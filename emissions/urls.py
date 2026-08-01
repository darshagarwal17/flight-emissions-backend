from django.urls import path
from .views import aircraft_leaderboard, geocode_location, live_flight_rate, live_flights, nearby_ngos, predict_emissions

urlpatterns = [
    path('predict-emissions/', predict_emissions),
    path('live-flights/', live_flights),
    path('live-flight-rate/', live_flight_rate),
    path('aircraft-leaderboard/', aircraft_leaderboard),
    path('nearby-ngos/', nearby_ngos),
    path('geocode/', geocode_location),
]