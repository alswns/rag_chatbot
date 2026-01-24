# 🎯 배포 & 운영 완성 가이드

## 📦 제공되는 파일 및 스크립트

### 🚀 시작/중지 스크립트

| 스크립트        | 목적                   | 사용법                     |
| --------------- | ---------------------- | -------------------------- |
| `start-rag.sh`  | CUDA 자동 감지 후 시작 | `./start-rag.sh`           |
| `manage-rag.sh` | 종합 관리 스크립트     | `./manage-rag.sh [명령어]` |

### 📚 가이드 문서

| 문서                   | 내용                            |
| ---------------------- | ------------------------------- |
| `QUICK_OPERATIONS.md`  | 일상 운영 명령어 (가장 자주 봄) |
| `DEPLOYMENT_GUIDE.md`  | 상세 배포 & 운영 가이드         |
| `LLM_BACKEND_GUIDE.md` | vLLM vs Ollama 설정 가이드      |
| `SETUP_COMPLETE.md`    | 초기 설정 완료 문서             |

---

## 🚀 가장 빠른 시작 방법

### 첫 번째 실행

```bash
cd /Users/bagminjun/Desktop/rag

# CUDA 자동 감지 후 시작
./start-rag.sh
```

**결과:**

```
✓ CUDA 감지 완료 (있으면 vLLM, 없으면 Ollama 자동 선택)
✓ 모든 컨테이너 시작
✓ 웹UI: http://localhost:3000
```

### 매일 시작/종료

```bash
# 아침: 시작
./start-rag.sh

# 저녁: 중지
./manage-rag.sh stop
```

---

## 📋 일일 운영 체크리스트

```bash
# 아침 (5분)
./start-rag.sh
./manage-rag.sh health-check

# 저녁 (1분)
./manage-rag.sh stop
```

---

## 🎯 주요 운영 명령어

### 상태 확인

```bash
./manage-rag.sh status         # 간단한 상태
./manage-rag.sh health-check   # 상세 헬스체크
```

### 로그 확인

```bash
./manage-rag.sh logs           # 모든 로그 (실시간)
./manage-rag.sh logs sync-worker  # 특정 서비스
```

### 백업/복구

```bash
./manage-rag.sh backup         # 백업 생성
./manage-rag.sh restore <파일> # 복구
```

### 유지보수

```bash
./manage-rag.sh restart        # 재시작
./manage-rag.sh update         # 이미지 업데이트
./manage-rag.sh clean          # 불필요한 파일 정리
```

---

## 🌍 배포 방식별 가이드

### 방식 1: 개발/테스트 (현재 상태)

```bash
cd /Users/bagminjun/Desktop/rag
./start-rag.sh
# http://localhost:3000 에서 접속
```

### 방식 2: 프로덕션 (Linux 서버)

```bash
# 1. 서버에 전송
scp -r rag/ user@server:/home/user/

# 2. 서버에서 실행
ssh user@server
cd /home/user/rag
nano .env  # 설정 수정

# 3. 시작
./start-rag.sh

# 또는 백그라운드 실행
nohup ./start-rag.sh > logs/startup.log 2>&1 &

# 4. 자동 시작 설정 (cron)
crontab -e
# 추가: @reboot /home/user/rag/start-rag.sh
```

### 방식 3: Docker Swarm (여러 서버)

```bash
# docker-stack.yml 사용
docker stack deploy -c docker-stack.yml rag
```

### 방식 4: Kubernetes (대규모)

```bash
# k8s manifest 파일 사용
kubectl apply -f k8s/
```

---

## 🔐 보안 체크리스트

배포 전 반드시 확인:

- [ ] `.env` 파일에 민감 정보 보관
- [ ] `.env` 파일을 `.gitignore`에 추가
- [ ] 포트는 로컬호스트 전용 (127.0.0.1:xxxx)
- [ ] 방화벽 설정 (필요한 포트만 개방)
- [ ] 정기 백업 설정
- [ ] 로그 모니터링 설정
- [ ] SSL/TLS 적용 (프로덕션)

---

## 💾 백업 전략

### 자동 백업 설정 (Linux)

```bash
# /etc/cron.d/rag-backup 생성
0 2 * * *  root  /Users/bagminjun/Desktop/rag/manage-rag.sh backup

# 또는 crontab에 추가
crontab -e
# 매일 밤 2시에 백업
0 2 * * * cd /Users/bagminjun/Desktop/rag && ./manage-rag.sh backup
```

### 백업 보존 정책

```bash
# 7일 이상 된 백업 삭제
find backups/ -name "*.tar.gz" -mtime +7 -delete

# 또는 스크립트로
cd /Users/bagminjun/Desktop/rag
find backups/ -name "*.tar.gz" -mtime +7 -exec rm {} \;
```

---

## 📊 모니터링 & 로깅

### 기본 모니터링

```bash
# 매일 아침 자동 헬스체크
0 9 * * * ./manage-rag.sh health-check >> logs/daily-check.log

# 주간 상태 리포트
0 9 * * 1 ./manage-rag.sh status >> logs/weekly-report.log
```

### 고급 모니터링 (선택사항)

```bash
# Prometheus + Grafana 연동
# docker-compose에 모니터링 스택 추가
```

---

## 🎓 운영자 가이드 (신규 담당자용)

### 첫날

1. `QUICK_OPERATIONS.md` 읽기 (10분)
2. `start-rag.sh` 실행해보기
3. 웹UI 접속 (http://localhost:3000)
4. 기본 명령어 시연

### 첫주

1. 로그 확인 방법 학습
2. 문제 해결 절차 학습
3. 백업/복구 테스트
4. 연락처 확인

### 첫달

1. 전체 운영 프로세스 숙지
2. 비상 상황 대응 연습
3. 정기 유지보수 실행
4. 문서 업데이트

---

## 🆘 긴급 상황 대응

### 서비스 완전 다운

```bash
# 1단계: 상태 확인
./manage-rag.sh health-check

# 2단계: 로그 확인
./manage-rag.sh logs | tail -100

# 3단계: 강제 재시작
docker-compose down
docker-compose up -d

# 4단계: 확인
./manage-rag.sh status
```

### 데이터 손실

```bash
# 최신 백업에서 복구
./manage-rag.sh restore backups/latest-backup.tar.gz

# 서비스 재시작
./manage-rag.sh restart
```

### 성능 저하

```bash
# 1. 리소스 정리
./manage-rag.sh clean

# 2. 오래된 로그 삭제
./manage-rag.sh prune-logs 7

# 3. 서비스 재시작
./manage-rag.sh restart
```

---

## 📞 기술 지원

### 문제 해결 순서

1. `QUICK_OPERATIONS.md`의 "문제 발생시" 섹션 확인
2. `DEPLOYMENT_GUIDE.md`의 "문제 해결" 섹션 참고
3. 로그에서 에러 메시지 확인
4. 관련 문서 검색

### 정보 수집

```bash
# 전체 시스템 정보 수집
docker-compose ps
docker system df
docker stats
df -h
```

---

## ✅ 배포 완료 확인

### 체크리스트

```
배포 전
- [ ] 하드웨어 요구사항 확인
- [ ] Docker/Docker Compose 설치
- [ ] .env 파일 준비

배포 중
- [ ] 저장소 다운로드
- [ ] 환경 설정
- [ ] 이미지 빌드
- [ ] 컨테이너 시작

배포 후
- [ ] 모든 컨테이너 실행중 확인
- [ ] 웹UI 접속 확인
- [ ] API 헬스체크 통과
- [ ] 백업 설정 확인
- [ ] 로그 모니터링 설정
- [ ] 문서 숙지
```

---

## 🎉 완료!

**RAG 시스템의 배포 및 운영 준비가 완료되었습니다!**

### 빠른 시작

```bash
cd /Users/bagminjun/Desktop/rag
./start-rag.sh
```

### 다음 문서 읽기

1. `QUICK_OPERATIONS.md` - 일일 운영
2. `DEPLOYMENT_GUIDE.md` - 상세 가이드
3. `LLM_BACKEND_GUIDE.md` - 성능 최적화

### 문의사항

로그를 확인하고 관련 가이드를 참고하세요!

---

**작성일**: 2026-01-25  
**상태**: ✅ 배포 준비 완료  
**담당자**: RAG 시스템 운영팀
