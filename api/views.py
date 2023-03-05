from rest_framework import status
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView, CreateAPIView, ListAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.settings import api_settings

from django.contrib.auth.models import User, AnonymousUser
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.conf import settings

from .utils import chatbot_response

class ChatView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        question = request.data['question']

        if question:
            response = chatbot_response(question)
            return Response({"answer":response}, status=status.HTTP_200_OK)
        else:
            return Response({"error":"No question provided"}, status=status.HTTP_400_BAD_REQUEST)

