from rest_framework import serializers


class ImageRetrievalSerializer(serializers.Serializer):
    folder_path = serializers.CharField()
    number_of_image_to_search = serializers.IntegerField()
    number_of_result = serializers.IntegerField()
