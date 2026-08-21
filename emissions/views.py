import os
import joblib
import pandas as pd
import requests
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from openap import FuelFlow
from .serializers import FlightInputSerializer

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'flight_emissions_model.pkl')
model = joblib.load(MODEL_PATH)

ALL_AIRCRAFT = ['A320', 'A321', 'A350', 'B737', 'B738', 'B777', 'B788']


@api_view(['POST'])
def predict_emissions(request):
    serializer = FlightInputSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    aircraft = serializer.validated_data['aircraft']
    distance_km = serializer.validated_data['distance_km']

    X_actual = pd.DataFrame([{'aircraft': aircraft, 'distance_km': distance_km}])
    actual_co2 = float(model.predict(X_actual)[0])

    X_all = pd.DataFrame([
        {'aircraft': ac, 'distance_km': distance_km} for ac in ALL_AIRCRAFT
    ])
    all_preds = model.predict(X_all)
    best_idx = all_preds.argmin()
    optimal_aircraft = ALL_AIRCRAFT[best_idx]
    optimal_co2 = float(all_preds[best_idx])

    return Response({
        'aircraft': aircraft,
        'distance_km': distance_km,
        'predicted_co2_kg': round(actual_co2, 2),
        'optimal_aircraft': optimal_aircraft,
        'optimal_co2_kg': round(optimal_co2, 2),
        'efficiency_gap_pct': round(((actual_co2 - optimal_co2) / actual_co2) * 100, 1)
    })


@api_view(['GET'])
def live_flights(request):
    lamin = request.GET.get('lamin', 45)
    lamax = request.GET.get('lamax', 55)
    lomin = request.GET.get('lomin', -5)
    lomax = request.GET.get('lomax', 15)

    url = "https://opensky-network.org/api/states/all"
    params = {'lamin': lamin, 'lamax': lamax, 'lomin': lomin, 'lomax': lomax}

    try:
        resp = requests.get(
            url,
            params=params,
            timeout=15,
            auth=(settings.OPENSKY_USERNAME, settings.OPENSKY_PASSWORD)
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException:
        return Response({
            'count': 0,
            'flights': [],
            'error': 'Live flight data is temporarily unavailable from this server.'
        }, status=200)

    flights = []
    for state in (data.get('states') or [])[:100]:
        icao24, callsign, origin_country = state[0], state[1], state[2]
        lon, lat, altitude = state[5], state[6], state[7]
        velocity, heading = state[9], state[10]

        if lat is None or lon is None:
            continue

        flights.append({
            'icao24': icao24,
            'callsign': (callsign or '').strip(),
            'origin_country': origin_country,
            'lat': lat,
            'lon': lon,
            'altitude_m': altitude,
            'velocity_ms': velocity,
            'heading': heading,
        })

    return Response({'count': len(flights), 'flights': flights})


@api_view(['POST'])
def live_flight_rate(request):
    aircraft = request.data.get('aircraft')
    altitude_m = request.data.get('altitude_m')
    velocity_ms = request.data.get('velocity_ms')

    if not aircraft or altitude_m is None or velocity_ms is None:
        return Response({'error': 'aircraft, altitude_m, and velocity_ms are required'}, status=400)

    altitude_ft = altitude_m * 3.28084
    speed_kt = velocity_ms * 1.94384

    try:
        fuelflow = FuelFlow(ac=aircraft)
        ff_kg_s = fuelflow.enroute(mass=60000, tas=speed_kt, alt=altitude_ft, vs=0)
    except Exception as e:
        return Response({'error': f'Could not compute for {aircraft}: {str(e)}'}, status=400)

    co2_kg_s = ff_kg_s * 3.16
    co2_kg_hr = co2_kg_s * 3600

    return Response({
        'aircraft': aircraft,
        'altitude_ft': round(altitude_ft),
        'speed_kt': round(speed_kt),
        'fuel_flow_kg_s': round(ff_kg_s, 3),
        'co2_kg_per_hour': round(co2_kg_hr, 1),
    })


@api_view(['GET'])
def aircraft_leaderboard(request):
    distance_km = 2000  # fixed reference distance for fair comparison
    X = pd.DataFrame([{'aircraft': ac, 'distance_km': distance_km} for ac in ALL_AIRCRAFT])
    preds = model.predict(X)

    results = sorted(
        [{'aircraft': ac, 'co2_kg': round(float(p), 1)} for ac, p in zip(ALL_AIRCRAFT, preds)],
        key=lambda r: r['co2_kg']
    )
    return Response({'distance_km': distance_km, 'aircraft': results})


@api_view(['GET'])
def nearby_ngos(request):
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return Response({'error': 'lat and lon are required'}, status=400)

    def run_query(query):
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={'data': query},
            headers={'User-Agent': 'FlightEmissionsProject/1.0 (student project)'},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    climate_query = f"""
    [out:json][timeout:25];
    (
      node["office"="environmental"](around:50000,{lat},{lon});
      node["office"="ngo"]["name"~"climate|carbon|environment|sustainab|green|eco",i](around:50000,{lat},{lon});
    );
    out center 15;
    """

    general_query = f"""
    [out:json][timeout:25];
    (
      node["office"="ngo"](around:50000,{lat},{lon});
      node["office"="environmental"](around:50000,{lat},{lon});
    );
    out center 15;
    """

    is_climate_specific = True
    try:
        data = run_query(climate_query)
        if not data.get('elements'):
            is_climate_specific = False
            data = run_query(general_query)
    except requests.exceptions.RequestException:
        return Response(
            {'ngos': [], 'error': 'The NGO data source is slow or unavailable right now. Try again shortly.'},
            status=200
        )

    results = []
    for el in data.get('elements', [])[:15]:
        tags = el.get('tags', {})
        name = tags.get('name')
        if not name:
            continue
        results.append({
            'name': name,
            'address': tags.get('addr:full') or tags.get('addr:street') or 'Address not listed',
            'website': tags.get('website') or tags.get('contact:website'),
        })

    return Response({'ngos': results, 'is_climate_specific': is_climate_specific})


@api_view(['GET'])
def geocode_location(request):
    query = request.GET.get('q')
    if not query:
        return Response({'error': 'q is required'}, status=400)

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={'q': query, 'format': 'json', 'limit': 1},
            headers={'User-Agent': 'FlightEmissionsProject/1.0 (student project)'},
            timeout=10
        )
        resp.raise_for_status()
        results = resp.json()
    except requests.exceptions.RequestException:
        return Response({'error': 'Geocoding service unavailable'}, status=200)

    if not results:
        return Response({'error': f'Could not find "{query}"'}, status=200)

    return Response({
        'display_name': results[0]['display_name'],
        'lat': float(results[0]['lat']),
        'lon': float(results[0]['lon']),
    })