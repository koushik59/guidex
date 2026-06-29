/**
 * maps.js — GuideX Navigation (Google Maps + saved locations)
 *
 * Map rendering & routing: Google Maps JavaScript API (Maps, Places, Directions)
 * Saved places:            SQLite on device, served by Flask /api/locations
 * Voice navigation:        polls /api/pending_nav (set by Vosk "navigate → 1/2/3")
 * TTS:                     /api/speak → backend espeak-ng on the Jetson
 */

// ── State ──────────────────────────────────────────────────────────────────────
let map = null;
let userMarker = null;
let directionsService = null;
let directionsRenderer = null;
let autocomplete = null;
let infoWindow = null;
let watchId = null;
let userLatLng = null;                // google.maps.LatLng
let mapVisible = true;
let locationsVisible = false;

let savedLocations = [];              // [{id, name, lat, lng}]
let locationMarkers = [];             // google.maps.Marker[]

let activeRouteSteps = [];
let activeStepIndex = 0;
let lastGuidanceAt = 0;
let activeDestinationName = '';

const STEP_ADVANCE_M = 35;
const GUIDANCE_REPEAT_MS = 30000;

// ── Boot ───────────────────────────────────────────────────────────────────────
window.addEventListener('load', function () {
    const poll = setInterval(function () {
        if (typeof google !== 'undefined' && google.maps) {
            clearInterval(poll);
            initMap();
        }
    }, 200);
});

function initMap() {
    map = new google.maps.Map(document.getElementById('googleMap'), {
        center: { lat: 20.5937, lng: 78.9629 },
        zoom: 15,
        mapTypeId: 'roadmap',
        styles: darkMapStyle(),
        zoomControl: true,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: true,
    });

    directionsService = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer({
        map: map,
        suppressMarkers: false,
        polylineOptions: { strokeColor: '#818cf8', strokeWeight: 5, strokeOpacity: 0.85 },
    });

    autocomplete = new google.maps.places.Autocomplete(
        document.getElementById('destinationInput'),
        { types: ['geocode'] }
    );

    infoWindow = new google.maps.InfoWindow();

    // Error banner inside map card
    const mapCard = document.getElementById('mapPanel');
    const errDiv = document.createElement('div');
    errDiv.id = 'mapErrorBanner';
    errDiv.style.cssText = [
        'display:none', 'align-items:flex-start', 'gap:10px',
        'padding:10px 14px', 'background:rgba(251,113,133,0.08)',
        'border-top:1px solid rgba(251,113,133,0.25)',
        'border-bottom:1px solid rgba(251,113,133,0.25)',
        'font-size:0.78rem', 'line-height:1.5', 'color:#fca5a5',
    ].join(';');
    mapCard.insertBefore(errDiv, document.getElementById('googleMap'));

    // Wire buttons
    document.getElementById('navigateBtn').addEventListener('click', function () { startMapNavigation(); });
    document.getElementById('saveLocationBtn').addEventListener('click', openSaveDialog);
    document.getElementById('toggleLocationsBtn').addEventListener('click', toggleLocationsPanel);
    document.getElementById('locateBtn').addEventListener('click', centerOnUser);
    document.getElementById('mapToggleBtn').addEventListener('click', toggleMap);
    document.getElementById('clearRouteBtn').addEventListener('click', clearRoute);
    document.getElementById('saveDlgConfirm').addEventListener('click', confirmSaveLocation);
    document.getElementById('saveDlgCancel').addEventListener('click', closeSaveDialog);
    document.getElementById('repeatStepBtn').addEventListener('click', function () {
        speakCurrentStep('Current direction');
    });
    document.getElementById('saveLocName').addEventListener('keydown', function (e) {
        if (e.key === 'Enter') confirmSaveLocation();
        if (e.key === 'Escape') closeSaveDialog();
    });
    const vnCancel = document.getElementById('voiceNameCancel');
    if (vnCancel) vnCancel.addEventListener('click', closeVoiceNameOverlay);

    startLocationWatch();
    loadSavedLocations();

    // Poll voice-triggered actions every 2 s
    setInterval(pollPendingNavigation, 2000);
    setInterval(pollSaveTrigger, 2000);
}

// ── Live Location ──────────────────────────────────────────────────────────────
function startLocationWatch() {
    if (!navigator.geolocation) {
        document.getElementById('mapCoords').textContent = 'GPS not supported';
        return;
    }
    watchId = navigator.geolocation.watchPosition(
        function (pos) {
            const lat = pos.coords.latitude;
            const lng = pos.coords.longitude;
            userLatLng = new google.maps.LatLng(lat, lng);

            document.getElementById('mapCoords').textContent =
                lat.toFixed(5) + ', ' + lng.toFixed(5);

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
                    zIndex: 999,
                });
                map.panTo(userLatLng);
            } else {
                userMarker.setPosition(userLatLng);
            }

            updateLiveGuidance();
        },
        function (err) {
            console.warn('[Maps] GPS error:', err.message);
            document.getElementById('mapCoords').textContent = 'Location unavailable';
        },
        { enableHighAccuracy: true, maximumAge: 5000, timeout: 10000 }
    );
}

function centerOnUser() {
    if (userLatLng && map) {
        map.panTo(userLatLng);
        map.setZoom(17);
    } else {
        showMapError('Location not available yet. Please allow location access.');
    }
}

// ── Saved Locations ────────────────────────────────────────────────────────────
async function loadSavedLocations() {
    try {
        const r = await fetch('/api/locations');
        savedLocations = await r.json();
    } catch (e) {
        savedLocations = [];
    }
    renderLocationMarkers();
    renderLocationsList();
    updatePlacesBadge();
}

function renderLocationMarkers() {
    locationMarkers.forEach(function (m) { m.setMap(null); });
    locationMarkers = [];

    savedLocations.forEach(function (loc) {
        const marker = new google.maps.Marker({
            position: { lat: loc.lat, lng: loc.lng },
            map: map,
            title: loc.name,
            label: {
                text: loc.name.charAt(0).toUpperCase(),
                color: '#07091a',
                fontSize: '11px',
                fontWeight: 'bold',
            },
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 13,
                fillColor: '#818cf8',
                fillOpacity: 1,
                strokeColor: '#ffffff',
                strokeWeight: 2,
            },
            zIndex: 100,
        });

        marker.addListener('click', function () {
            infoWindow.setContent(
                '<div class="gmap-popup">' +
                '<div class="popup-name">' + escHtml(loc.name) + '</div>' +
                '<button class="popup-btn popup-nav" ' +
                '        onclick="navigateToSavedLocation(' + loc.id + ');' +
                '                 window.gmapsInfoWindow && window.gmapsInfoWindow.close()">Navigate here</button>' +
                '<button class="popup-btn popup-del" ' +
                '        onclick="deleteLocation(' + loc.id + ');' +
                '                 window.gmapsInfoWindow && window.gmapsInfoWindow.close()">Delete</button>' +
                '</div>'
            );
            infoWindow.open(map, marker);
            window.gmapsInfoWindow = infoWindow;
        });

        locationMarkers.push(marker);
    });
}

function renderLocationsList() {
    const list = document.getElementById('locationsList');
    if (!list) return;
    if (!savedLocations.length) {
        list.innerHTML =
            '<div class="loc-empty">No saved locations yet.<br>' +
            'Press <strong>Save Location</strong> or say <em>"save"</em>.</div>';
        return;
    }
    list.innerHTML = savedLocations.map(function (loc, i) {
        return (
            '<div class="loc-item" data-id="' + loc.id + '">' +
            '<div class="loc-num">' + (i + 1) + '</div>' +
            '<div class="loc-info">' +
            '<div class="loc-name">' + escHtml(loc.name) + '</div>' +
            '<div class="loc-coord">' + loc.lat.toFixed(4) + ', ' + loc.lng.toFixed(4) + '</div>' +
            '</div>' +
            '<div class="loc-btns">' +
            '<button class="loc-nav-btn" onclick="navigateToSavedLocation(' + loc.id + ')" title="Navigate">&#x2192;</button>' +
            '<button class="loc-del-btn" onclick="deleteLocation(' + loc.id + ')" title="Delete">&#x2715;</button>' +
            '</div>' +
            '</div>'
        );
    }).join('');
}

function updatePlacesBadge() {
    const btn = document.getElementById('toggleLocationsBtn');
    if (!btn) return;
    const n = savedLocations.length;
    btn.textContent = (locationsVisible ? 'Hide Places' : 'My Places') + (n ? ' (' + n + ')' : '');
}

// ── Save Location Dialog ───────────────────────────────────────────────────────
function openSaveDialog() {
    if (!userLatLng) {
        mapSpeak('GPS not available. Cannot save location.');
        return;
    }
    document.getElementById('saveLocName').value = '';
    document.getElementById('saveLocCoords').textContent =
        userLatLng.lat().toFixed(5) + ', ' + userLatLng.lng().toFixed(5);
    document.getElementById('saveLocDialog').style.display = 'flex';
    setTimeout(function () { document.getElementById('saveLocName').focus(); }, 80);
}

function closeSaveDialog() {
    document.getElementById('saveLocDialog').style.display = 'none';
}

async function confirmSaveLocation() {
    const name = document.getElementById('saveLocName').value.trim();
    if (!name) { document.getElementById('saveLocName').focus(); return; }
    if (!userLatLng) { closeSaveDialog(); return; }

    try {
        const r = await fetch('/api/locations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                lat: userLatLng.lat(),
                lng: userLatLng.lng(),
            }),
        });
        if (r.ok) {
            closeSaveDialog();
            await loadSavedLocations();
        }
    } catch (e) {
        console.error('[Maps] Save failed:', e);
    }
}

async function deleteLocation(id) {
    const loc = savedLocations.find(function (l) { return l.id === id; });
    if (!confirm('Delete "' + (loc ? loc.name : 'this location') + '"?')) return;
    try {
        await fetch('/api/locations/' + id, { method: 'DELETE' });
        await loadSavedLocations();
    } catch (e) {
        console.error('[Maps] Delete failed:', e);
    }
}

// ── Navigate to saved location ─────────────────────────────────────────────────
function navigateToSavedLocation(id) {
    const loc = savedLocations.find(function (l) { return l.id === id; });
    if (!loc) return;
    if (!userLatLng) {
        mapSpeak('GPS not available. Cannot start navigation.');
        return;
    }
    activeDestinationName = loc.name;
    document.getElementById('destinationInput').value = loc.name;

    const request = {
        origin: userLatLng,
        destination: new google.maps.LatLng(loc.lat, loc.lng),
        travelMode: google.maps.TravelMode.WALKING,
    };

    directionsService.route(request, function (result, status) {
        if (status === 'OK') {
            clearMapError();
            directionsRenderer.setDirections(result);

            const leg = result.routes[0].legs[0];
            activeRouteSteps = leg.steps || [];
            activeStepIndex = 0;
            lastGuidanceAt = 0;

            document.getElementById('routeDestName').textContent = loc.name;
            document.getElementById('routeDistance').textContent = leg.distance.text;
            document.getElementById('routeDuration').textContent = leg.duration.text;
            document.getElementById('routeInfo').style.display = 'flex';

            const summary = 'Route to ' + loc.name + '. ' + leg.distance.text +
                            ', about ' + leg.duration.text + ' walking.';
            mapSpeak(summary);
            setTimeout(function () { speakCurrentStep('First direction'); }, 2200);
        } else {
            handleDirectionsError(status);
        }
    });
}

// ── Navigate to typed / autocomplete destination ───────────────────────────────
function startMapNavigation(destinationOverride) {
    if (!userLatLng) {
        mapSpeak('Your location is not available yet. Please wait.');
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
        const inputText = document.getElementById('destinationInput').value.trim();
        if (!inputText) {
            mapSpeak('Please enter a destination or choose a saved place.');
            return;
        }
        destination = inputText;
        activeDestinationName = inputText;
    }

    directionsService.route(
        { origin: userLatLng, destination: destination, travelMode: google.maps.TravelMode.WALKING },
        function (result, status) {
            if (status === 'OK') {
                clearMapError();
                directionsRenderer.setDirections(result);

                const leg = result.routes[0].legs[0];
                activeRouteSteps = leg.steps || [];
                activeStepIndex = 0;
                lastGuidanceAt = 0;

                document.getElementById('routeDestName').textContent = activeDestinationName;
                document.getElementById('routeDistance').textContent = leg.distance.text;
                document.getElementById('routeDuration').textContent = leg.duration.text;
                document.getElementById('routeInfo').style.display = 'flex';

                mapSpeak('Route to ' + activeDestinationName + ' found. ' +
                         leg.distance.text + ', about ' + leg.duration.text + ' walking.');
                setTimeout(function () { speakCurrentStep('First direction'); }, 2200);
            } else {
                handleDirectionsError(status);
            }
        }
    );
}

function handleDirectionsError(status) {
    if (status === 'REQUEST_DENIED') {
        showMapError('Directions API not enabled or billing not active on this API key.');
        mapSpeak('Navigation unavailable. Directions API not enabled.');
    } else if (status === 'ZERO_RESULTS') {
        showMapError('No walking route found to that destination.');
        mapSpeak('No route found. Try a different destination.');
    } else if (status === 'NOT_FOUND') {
        showMapError('Destination not found. Check the address.');
        mapSpeak('Destination not found.');
    } else {
        showMapError('Directions error: ' + status);
        mapSpeak('Could not find a route. Please try again.');
    }
}

// ── Clear Route ────────────────────────────────────────────────────────────────
function clearRoute() {
    directionsRenderer.setDirections({ routes: [] });
    activeRouteSteps = [];
    activeStepIndex = 0;
    activeDestinationName = '';
    lastGuidanceAt = 0;
    document.getElementById('routeInfo').style.display = 'none';
    document.getElementById('destinationInput').value = '';
}

// ── Turn-by-Turn Guidance ──────────────────────────────────────────────────────
function stripHtml(html) {
    const tmp = document.createElement('div');
    tmp.innerHTML = html || '';
    return tmp.textContent || tmp.innerText || '';
}

function speakCurrentStep(prefix) {
    prefix = prefix || 'Next direction';
    if (!activeRouteSteps.length || activeStepIndex >= activeRouteSteps.length) {
        mapSpeak('No active directions.');
        return;
    }
    const step = activeRouteSteps[activeStepIndex];
    const instruction = stripHtml(step.instructions);
    const dist = step.distance ? step.distance.text : '';
    mapSpeak(prefix + '. In ' + dist + ', ' + instruction + '.');

    const el = document.getElementById('currentStep');
    if (el) el.textContent = instruction + (dist ? ' (' + dist + ')' : '');

    lastGuidanceAt = Date.now();
}

function getDistanceMeters(from, to) {
    if (!from || !to) return Infinity;
    const lat1 = from.lat() * Math.PI / 180;
    const lat2 = to.lat() * Math.PI / 180;
    const dp   = lat2 - lat1;
    const dl   = (to.lng() - from.lng()) * Math.PI / 180;
    const a    = Math.sin(dp / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dl / 2) ** 2;
    return 6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function updateLiveGuidance() {
    if (!activeRouteSteps.length || !userLatLng || activeStepIndex >= activeRouteSteps.length) return;
    const step = activeRouteSteps[activeStepIndex];
    const dist = getDistanceMeters(userLatLng, step.end_location);
    const now  = Date.now();

    if (dist < STEP_ADVANCE_M && activeStepIndex < activeRouteSteps.length - 1) {
        activeStepIndex++;
        speakCurrentStep('Now');
        return;
    }
    if (now - lastGuidanceAt > GUIDANCE_REPEAT_MS) {
        speakCurrentStep('Continue');
    }
}

// ── Voice Nav Polling ──────────────────────────────────────────────────────────
async function pollPendingNavigation() {
    try {
        const r = await fetch('/api/pending_nav');
        const d = await r.json();
        if (d.navigation) {
            await loadSavedLocations();         // ensure list is fresh
            navigateToSavedLocation(d.navigation.id);
        }
    } catch (e) { /* ignore */ }
}

async function pollSaveTrigger() {
    try {
        const r = await fetch('/api/save_triggered');
        const d = await r.json();
        if (d.triggered) openVoiceNameOverlay();
    } catch (e) { /* ignore */ }
}

// ── Voice name capture overlay ──────────────────────────────────────────────────
let voiceNamePollTimer = null;

function openVoiceNameOverlay() {
    const ov = document.getElementById('voiceNameOverlay');
    if (!ov) return;
    document.getElementById('voiceNameResult').textContent = '';
    ov.style.display = 'flex';
    // Poll backend for the captured name every 800 ms
    if (voiceNamePollTimer) clearInterval(voiceNamePollTimer);
    voiceNamePollTimer = setInterval(async function () {
        try {
            const r = await fetch('/api/voice_name');
            const d = await r.json();
            if (d.name) {
                clearInterval(voiceNamePollTimer);
                voiceNamePollTimer = null;
                document.getElementById('voiceNameResult').textContent = '"' + d.name + '"';
                // Small pause so the user can see the name, then auto-save
                setTimeout(function () { finishVoiceSave(d.name); }, 900);
            }
        } catch (e) { /* ignore */ }
    }, 800);
}

function closeVoiceNameOverlay() {
    const ov = document.getElementById('voiceNameOverlay');
    if (ov) ov.style.display = 'none';
    if (voiceNamePollTimer) { clearInterval(voiceNamePollTimer); voiceNamePollTimer = null; }
}

async function finishVoiceSave(name) {
    closeVoiceNameOverlay();
    if (!userLatLng) {
        mapSpeak('GPS not available yet. Please wait and try again.');
        return;
    }
    try {
        const r = await fetch('/api/locations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, lat: userLatLng.lat(), lng: userLatLng.lng() }),
        });
        if (r.ok) {
            await loadSavedLocations();
            mapSpeak('Location saved as ' + name + '.');
        } else {
            mapSpeak('Could not save location.');
        }
    } catch (e) {
        mapSpeak('Could not save location.');
    }
}

// ── UI helpers ─────────────────────────────────────────────────────────────────
function showMapError(html) {
    const b = document.getElementById('mapErrorBanner');
    if (!b) return;
    b.innerHTML = '<span>⚠️</span><span>' + html + '</span>';
    b.style.display = 'flex';
}
function clearMapError() {
    const b = document.getElementById('mapErrorBanner');
    if (b) b.style.display = 'none';
}

function toggleMap() {
    const panel = document.getElementById('mapPanel');
    mapVisible = !mapVisible;
    panel.style.display = mapVisible ? 'flex' : 'none';
    const btn = document.getElementById('mapToggleBtn');
    if (btn) btn.querySelector('span:last-child').textContent = mapVisible ? 'Hide Map' : 'Show Map';
}

function toggleLocationsPanel() {
    locationsVisible = !locationsVisible;
    const panel = document.getElementById('locationsPanel');
    panel.style.display = locationsVisible ? 'flex' : 'none';
    updatePlacesBadge();
    if (locationsVisible) renderLocationsList();
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── TTS ────────────────────────────────────────────────────────────────────────
// Web Speech API is primary — works in browser on both laptop and Jetson.
// Backend espeak is also fired so the Jetson's physical speaker gets audio
// even when the browser is backgrounded, but its failure is always silent.
function mapSpeak(text) {
    if (window.speechSynthesis) {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 0.92; u.volume = 1;
        speechSynthesis.speak(u);
    }
    // Fire-and-forget to backend espeak (Jetson hardware speaker)
    fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text }),
    }).catch(function () { /* espeak not available on this platform — ignore */ });
}

// ── Public API ─────────────────────────────────────────────────────────────────
window.guideXMaps = {
    navigateTo:     startMapNavigation,
    navigateToSaved: navigateToSavedLocation,
    clearRoute:     clearRoute,
    repeatGuidance: function () { speakCurrentStep('Current direction'); },
    whereAmI: function () {
        if (userLatLng) {
            mapSpeak('Your location is latitude ' + userLatLng.lat().toFixed(4) +
                     ', longitude ' + userLatLng.lng().toFixed(4) + '.');
        } else {
            mapSpeak('Location not available yet.');
        }
    },
    saveLocation:     openSaveDialog,
    refreshLocations: loadSavedLocations,
};

// ── Dark Map Style ─────────────────────────────────────────────────────────────
function darkMapStyle() {
    return [
        { elementType: 'geometry',             stylers: [{ color: '#1a1a2e' }] },
        { elementType: 'labels.text.stroke',   stylers: [{ color: '#1a1a2e' }] },
        { elementType: 'labels.text.fill',     stylers: [{ color: '#a0a0b0' }] },
        { featureType: 'road', elementType: 'geometry',             stylers: [{ color: '#16213e' }] },
        { featureType: 'road', elementType: 'geometry.stroke',      stylers: [{ color: '#212a37' }] },
        { featureType: 'road', elementType: 'labels.text.fill',     stylers: [{ color: '#9ca5b3' }] },
        { featureType: 'road.highway', elementType: 'geometry',     stylers: [{ color: '#0f3460' }] },
        { featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{ color: '#1f2835' }] },
        { featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{ color: '#f3d19c' }] },
        { featureType: 'water',     elementType: 'geometry',        stylers: [{ color: '#0e1626' }] },
        { featureType: 'water',     elementType: 'labels.text.fill', stylers: [{ color: '#515c6d' }] },
        { featureType: 'poi',       elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
        { featureType: 'poi.park',  elementType: 'geometry',        stylers: [{ color: '#263c3f' }] },
        { featureType: 'poi.park',  elementType: 'labels.text.fill', stylers: [{ color: '#6b9a76' }] },
        { featureType: 'transit',   elementType: 'geometry',        stylers: [{ color: '#2f3948' }] },
        { featureType: 'administrative', elementType: 'geometry.stroke', stylers: [{ color: '#4b6878' }] },
    ];
}
