from rest_framework import serializers

class FlightInputSerializer(serializers.Serializer):
    aircraft = serializers.CharField()
    distance_km = serializers.FloatField()