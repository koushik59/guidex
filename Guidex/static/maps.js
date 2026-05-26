/**
 * maps.js — Google Maps Integration for GuideX Navigation
 * Handles: Live location, Places autocomplete, Walking directions, Route info
 */

let map = null;
let userMarker = null;
let directionsService = null;
let directionsRenderer = null;
let autocomplete = null;
let watchId = null;
let userLatLng = null;
let mapVisible = true;
let activeRouteSteps = [];
let activeStepIndex = 0;
let lastGuidanceAt = 0;
let activeDestinationName = '';

// ── Wait for Google Maps API to load ──────────────────────────────────────────
window.addEventListener('load', function () {
    // Poll until google.maps is available (loaded async)
    const interval = setInterval(() => {
        if (typeof google !== 'undefined' && google.maps) {
            clearInterval(interval);
            initMap();
        }
    }, 200);
});

// ── Initialize Map ─────────────────────────────────────────────────────────────
function initMap() {
    // Default center: India (will update to user location)
    const defaultCenter = { lat: 20.5937, lng: 78.9629 };

    map = new google.maps.Map(document.getElementById('googleMap'), {
        center: defaultCenter,
        zoom: 15,
        mapTypeId: 'roadmap',
        styles: darkMapStyle(),          // Dark theme to match UI
        disableDefaultUI: false,
        zoomControl: true,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: true,
    });

    directionsService = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer({
        map: map,
        suppressMarkers: false,
        polylineOptions: {
            strokeColor: '#818cf8',
            strokeWeight: 5,
            strokeOpacity: 0.85,
        },
    });

    // Setup Places Autocomplete on the input field
    autocomplete = new google.maps.places.Autocomplete(
        document.getElementById('destinationInput'),
        { types: ['geocode'] }
    );

    // Start watching user location
    startLocationWatch();

    // Wire up buttons
    document.getElementById('navigateBtn').addEventListener('click', () => startMapNavigation());
    document.getElementById('locateBtn').addEventListener('click', centerOnUser);
    document.getElementById('clearRouteBtn').addEventListener('click', clearRoute);
    document.getElementById('mapToggleBtn').addEventListener('click', toggleMap);

    // Inject error banner element into map card
    const mapCard = document.getElementById('mapPanel');
    const errDiv = document.createElement('div');
    errDiv.id = 'mapErrorBanner';
    errDiv.style.cssText = [
        'display:none', 'align-items:flex-start', 'gap:10px',
        'padding:10px 14px', 'margin:0',
        'background:rgba(251,113,133,0.08)',
        'border-top:1px solid rgba(251,113,133,0.25)',
        'border-bottom:1px solid rgba(251,113,133,0.25)',
        'font-size:0.78rem', 'line-height:1.5',
        'color:#fca5a5',
    ].join(';');
    // Insert banner just above the map div
    const mapDiv = document.getElementById('googleMap');
    mapCard.insertBefore(errDiv, mapDiv);
}

// ── Live Location Tracking ─────────────────────────────────────────────────────
function startLocationWatch() {
    if (!navigator.geolocation) {
        document.getElementById('mapCoords').textContent = 'Geolocation not supported';
        return;
    }

    watchId = navigator.geolocation.watchPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            userLatLng = new google.maps.LatLng(lat, lng);

            // Update coordinate display
            document.getElementById('mapCoords').textContent =
                `${lat.toFixed(5)}, ${lng.toFixed(5)}`;

            // Place / update user marker
            if (!userMarker) {
                userMarker = new google.maps.Marker({
                    position: userLatLng,
                    map: map,
                    title: 'You are here',
                    icon: {
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 10,
                        fillColor: '#818cf8',
                        fillOpacity: 1,
                        strokeColor: '#ffffff',
                        strokeWeight: 2,
                    },
                });
                map.panTo(userLatLng);
            } else {
                userMarker.setPosition(userLatLng);
            }

            updateLiveGuidance();
        },
        (error) => {
            console.warn('Geolocation error:', error.message);
            document.getElementById('mapCoords').textContent = 'Location unavailable';
        },
        { enableHighAccuracy: true, maximumAge: 5000, timeout: 10000 }
    );
}

// ── Center map on user ─────────────────────────────────────────────────────────
function centerOnUser() {
    if (userLatLng && map) {
        map.panTo(userLatLng);
        map.setZoom(17);
    } else {
        showMapError('⚠️ Location not available yet. Please allow location access in your browser.');
    }
}

// ── Show / Clear error banner ──────────────────────────────────────────────────
function showMapError(html) {
    const banner = document.getElementById('mapErrorBanner');
    if (!banner) return;
    banner.innerHTML = `<span style="font-size:1.1rem">⚠️</span><span>${html}</span>`;
    banner.style.display = 'flex';
}

function clearMapError() {
    const banner = document.getElementById('mapErrorBanner');
    if (banner) banner.style.display = 'none';
}

// ── Start Navigation (Directions) ──────────────────────────────────────────────
function startMapNavigation(destinationOverride = null) {
    if (!userLatLng) {
        speak('Your location is not available yet. Please wait.');
        return;
    }

    const place = autocomplete ? autocomplete.getPlace() : null;
    let destination;

    if (destinationOverride) {
        destination = destinationOverride;
        activeDestinationName = destinationOverride;
        document.getElementById('destinationInput').value = destinationOverride;
    } else if (place && place.geometry) {
        destination = place.geometry.location;
        activeDestinationName = place.name || document.getElementById('destinationInput').value.trim();
    } else {
        // Fallback: use raw text input
        const inputText = document.getElementById('destinationInput').value.trim();
        if (!inputText) {
            speak('Please enter or say a destination first.');
            return;
        }
        destination = inputText;
        activeDestinationName = inputText;
    }

    const request = {
        origin: userLatLng,
        destination: destination,
        travelMode: google.maps.TravelMode.WALKING,  // WALKING suits blind navigation
    };

    directionsService.route(request, (result, status) => {
        if (status === 'OK') {
            clearMapError();
            directionsRenderer.setDirections(result);

            // Extract route summary
            const leg = result.routes[0].legs[0];
            activeRouteSteps = leg.steps || [];
            activeStepIndex = 0;
            lastGuidanceAt = 0;

            document.getElementById('routeDistance').textContent = `Distance: ${leg.distance.text}`;
            document.getElementById('routeDuration').textContent = `Duration: ${leg.duration.text}`;
            document.getElementById('routeInfo').style.display = 'flex';

            // Voice announce
            speak(`Route to ${activeDestinationName || 'your destination'} found. Distance: ${leg.distance.text}. Estimated time: ${leg.duration.text}.`);
            setTimeout(() => speakCurrentStep('First direction'), 1800);

        } else if (status === 'REQUEST_DENIED') {
            showMapError(
                '<strong>Directions API not enabled.</strong> ' +
                'Go to <a href="https://console.cloud.google.com/apis/library/directions-backend.googleapis.com" ' +
                'target="_blank" style="color:#818cf8">Google Cloud Console</a>, ' +
                'enable the <strong>Directions API</strong>, and make sure <strong>billing is active</strong> on your project.'
            );
            speak('Navigation request was denied. Please enable the Directions API in Google Cloud Console.');

        } else if (status === 'ZERO_RESULTS') {
            showMapError('⚠️ No walking route found to that destination. Try a different address.');
            speak('No route found. Please try a different destination.');

        } else if (status === 'NOT_FOUND') {
            showMapError('⚠️ Destination not found. Please check the address and try again.');
            speak('Destination not found.');

        } else {
            console.error('Directions error:', status);
            showMapError(`Directions error: ${status}. Please try again.`);
            speak('Could not find a route. Please try a different destination.');
        }
    });
}

// ── Clear Route ────────────────────────────────────────────────────────────────
function clearRoute() {
    directionsRenderer.setDirections({ routes: [] });
    document.getElementById('routeInfo').style.display = 'none';
    document.getElementById('destinationInput').value = '';
    activeRouteSteps = [];
    activeStepIndex = 0;
    activeDestinationName = '';
    lastGuidanceAt = 0;
}

function stripInstruction(html) {
    const temp = document.createElement('div');
    temp.innerHTML = html || '';
    return temp.textContent || temp.innerText || '';
}

function getDistanceMeters(fromLatLng, toLatLng) {
    if (!fromLatLng || !toLatLng) return Infinity;

    const lat1 = fromLatLng.lat() * Math.PI / 180;
    const lat2 = toLatLng.lat() * Math.PI / 180;
    const dLat = lat2 - lat1;
    const dLng = (toLatLng.lng() - fromLatLng.lng()) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
    return 6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function speakCurrentStep(prefix = 'Next direction') {
    if (!activeRouteSteps.length || activeStepIndex >= activeRouteSteps.length) {
        speak('No active route directions are available.');
        return;
    }

    const step = activeRouteSteps[activeStepIndex];
    const instruction = stripInstruction(step.instructions);
    const distance = step.distance ? step.distance.text : '';
    speak(`${prefix}. In ${distance}, ${instruction}.`);
    lastGuidanceAt = Date.now();
}

function updateLiveGuidance() {
    if (!activeRouteSteps.length || !userLatLng || activeStepIndex >= activeRouteSteps.length) {
        return;
    }

    const step = activeRouteSteps[activeStepIndex];
    const distanceToStepEnd = getDistanceMeters(userLatLng, step.end_location);
    const now = Date.now();

    if (distanceToStepEnd < 35 && activeStepIndex < activeRouteSteps.length - 1) {
        activeStepIndex += 1;
        speakCurrentStep('Now');
        return;
    }

    if (now - lastGuidanceAt > 30000) {
        speakCurrentStep('Continue');
    }
}

function repeatMapGuidance() {
    speakCurrentStep('Current direction');
}

function speakMapLocation() {
    const coords = document.getElementById('mapCoords').textContent;
    if (userLatLng) {
        speak(`Your current location is latitude ${userLatLng.lat().toFixed(5)}, longitude ${userLatLng.lng().toFixed(5)}.`);
    } else {
        speak(`Location is not available yet. ${coords}`);
    }
}

// ── Toggle Map Panel Visibility ────────────────────────────────────────────────
function toggleMap() {
    const panel = document.getElementById('mapPanel');
    mapVisible = !mapVisible;
    panel.style.display = mapVisible ? 'flex' : 'none';
    const btn = document.getElementById('mapToggleBtn');
    btn.querySelector('span:last-child').textContent = mapVisible ? 'Hide Map' : 'Show Map';
}

// ── Voice Helper (uses existing TTS from script.js if available) ───────────────
function speak(text) {
    if (window.speechSynthesis) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.volume = 1;
        speechSynthesis.cancel();
        speechSynthesis.speak(utterance);
    }
}

window.guideXMaps = {
    navigateTo: startMapNavigation,
    repeatGuidance: repeatMapGuidance,
    whereAmI: speakMapLocation,
    clearRoute: clearRoute,
};

// ── Dark Map Theme ─────────────────────────────────────────────────────────────
function darkMapStyle() {
    return [
        { elementType: 'geometry', stylers: [{ color: '#1a1a2e' }] },
        { elementType: 'labels.text.stroke', stylers: [{ color: '#1a1a2e' }] },
        { elementType: 'labels.text.fill', stylers: [{ color: '#a0a0b0' }] },
        { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#16213e' }] },
        { featureType: 'road', elementType: 'geometry.stroke', stylers: [{ color: '#212a37' }] },
        { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#9ca5b3' }] },
        { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#0f3460' }] },
        { featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{ color: '#1f2835' }] },
        { featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{ color: '#f3d19c' }] },
        { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#0e1626' }] },
        { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#515c6d' }] },
        { featureType: 'poi', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
        { featureType: 'poi.park', elementType: 'geometry', stylers: [{ color: '#263c3f' }] },
        { featureType: 'poi.park', elementType: 'labels.text.fill', stylers: [{ color: '#6b9a76' }] },
        { featureType: 'transit', elementType: 'geometry', stylers: [{ color: '#2f3948' }] },
        { featureType: 'transit.station', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
        { featureType: 'administrative', elementType: 'geometry.stroke', stylers: [{ color: '#4b6878' }] },
        { featureType: 'administrative.land_parcel', elementType: 'labels.text.fill', stylers: [{ color: '#64779e' }] },
        { featureType: 'administrative.province', elementType: 'geometry.stroke', stylers: [{ color: '#4b6878' }] },
    ];
}
