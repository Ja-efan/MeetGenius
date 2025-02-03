from django.shortcuts import render, get_object_or_404
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes, authentication_classes

from rest_framework.response import Response
from rest_framework import status
from .models import Project, ProjectParticipation, Report, Document
from .serializers import ProjectSerializer
from rest_framework.permissions import IsAuthenticatedOrReadOnly,AllowAny,IsAuthenticated

from docx import Document as DocxDocument  # Word 문서 읽기용
import os # 파일 확장자 처리리


# Create your views here.

User = get_user_model()


@api_view(['GET','POST'])
@permission_classes([IsAuthenticated]) # 인증되지 않은 사용자는 접근 불가
def project_list_create(request):
    """
    프록젝트 목록 조회 및 생성 API
    - 'GET' : 전체 목록 반환
    - 'POST' : 프로젝트 생성 및 참여자 추가
    """
    if request.method == "GET":
        projects = Project.objects.all()
        serializer = ProjectSerializer(projects, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    # 프로젝트 생성 및 참여자 추가
    elif request.method == "POST":
        serializer = ProjectSerializer(data=request.data)
        if serializer.is_valid():
            creator = request.user if request.user.is_authenticated else None # 로그인 구현되면 수정하기.
            department_id = request.data.get("department")
            project = serializer.save(creator = creator,
                                      department_id = department_id
                                      ) # 현재 로그인한 유저가 생성자로 저장
            
            participants = request.data.get("participants", [])
            for participant in participants:
                user_id = participant.get("id")
                authority = participant.get("authority", 1) 

                user= get_object_or_404(User, id=user_id)

                ProjectParticipation.objects.create(
                    project=project, participant = user, authority = authority
                )
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(
            {"status": "error", "message": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )
    

@api_view(['PUT', 'PATCH'])
@permission_classes([IsAuthenticated]) # 인증되지 않은 사용자는 접근 불가
def project_update(request, project_id):
    
    """
    특정 프로젝트 내용 수정하는 API
    - 'PUT' : 전체 필드 업데이트
    - 'PATCH' : 일부 필드 업데이트 -> 대부분의 경우에 얘가 더 유연한 듯.
    """
    project = get_object_or_404(Project, id=project_id)

    serializer = ProjectSerializer(project, data=request.data, partial =(request.method =='PATCH'))

    if serializer.is_valid():
        serializer.save() # Django의 ManyToManyField는 이 메소드만으로는 업데이트 되지 않음.

        if "participants" in request.data:
            participants = request.data.get("participants", []) # 리스트로 받아옴
            project.participants.all().delete() # 기존 참여자 제거

            for participant in participants:
                user_id = participant.get("id")
                authority = participant.get("authority",1)

                user = get_object_or_404(User, id=user_id)
                ProjectParticipation.objects.create(
                    project=project,participant=user,authority=authority
                )

        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response({'status':'error','message':serializer.errors},status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_report(request, project_id):
    '''
    보고서를 저장하는 API
        파일을 저장하지 않고 즉시 내용을 읽어 'Report 모델에 저장'
        0. 파일 내용 뽑기.
        1. Document 생성 (type=2)
        2. Report 생성 + Document와 연결
    '''
    project = get_object_or_404(Project, id = project_id)
    department = project.department
    uploaded_file = request.FILES.get('file') # 파일 받기
    doc_type = 2
    print(project_id)
    if not uploaded_file:
        print('No file')
        return Response({'error': 'no file'},status=status.HTTP_400_BAD_REQUEST)
    
    title = os.path.splitext(uploaded_file.name)[0]

    try : 
        if uploaded_file.name.endswith('.docx'):
            doc = DocxDocument(uploaded_file)
            file_content = "\n".join([p.text for p in doc.paragraphs])  # 문서의 모든 텍스트 합치기
        else : # 일반 txt
            file_content = uploaded_file.read().decode('utf-8')
    except UnicodeEncodeError:
        return Response({"error": "파일 인코딩 오류. UTF-8 형식이어야 합니다."}, status=status.HTTP_400_BAD_REQUEST)
    print(file_content)
    # Document 객체 생성
    document = Document.objects.create(
        type=doc_type,
        project=project,
        department=department)
    
    # Report 객체 생성
    report = Report.objects.create(
        document = document,
        project = project,
        writer = request.user,
        title = title,
        content = file_content,
    )
    return Response({"message": "파일 내용 저장 완료", "report_id": report.id}, status=status.HTTP_201_CREATED)
    
