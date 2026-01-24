# 🚀 RAG 시스템 - 빠른 운영 가이드

## 📌 가장 자주 사용하는 명령어

### 일일 운영

```bash
# 1️⃣ 서비스 시작 (매일 아침)
cd /Users/bagminjun/Desktop/rag
./start-rag.sh

# 또는 관리 스크립트 사용
./manage-rag.sh start

# 2️⃣ 서비스 상태 확인
./manage-rag.sh status

# 3️⃣ 로그 확인 (문제 발생시)
./manage-rag.sh logs
./manage-rag.sh logs sync-worker  # 특정 서비스

# 4️⃣ 서비스 중지 (퇴근전)
./manage-rag.sh stop
```

---

## 📋 정리된 운영 체크리스트

### 매일

- [ ] 서비스 시작: `./start-rag.sh`
- [ ] 상태 확인: `./manage-rag.sh status`
- [ ] 문제 없는지 확인
- [ ] 퇴근전 서비스 중지: `./manage-rag.sh stop`

### 주간 (금요일)

```bash
# 1. 전체 상태 확인
./manage-rag.sh health-check

# 2. 백업
./manage-rag.sh backup

# 3. 로그 검토
./manage-rag.sh logs | tail -100
```

### 월간 (말일)

```bash
# 1. 시스템 정리
./manage-rag.sh clean

# 2. 오래된 로그 삭제
./manage-rag.sh prune-logs 30

# 3. 이미지 업데이트
./manage-rag.sh update

# 4. 전체 백업
./manage-rag.sh backup
```

---

## 🔍 웹 서비스 접속

### 개발/테스트

```
http://localhost:3000   # Open WebUI (채팅 인터페이스)
http://localhost:9000   # RAG API (개발자용)
```

### 프로덕션 (원격)

```
http://<서버-IP>:3000   # Open WebUI
```

> ⚠️ 로컬호스트만 접근 가능하도록 설정됨 (보안)
> 원격 접근 필요시: docker-compose.yml의 포트 설정 변경

---

## 🚨 문제 발생시

### 빠른 해결 순서

```bash
# 1단계: 상태 확인
./manage-rag.sh health-check

# 2단계: 로그 확인
./manage-rag.sh logs

# 3단계: 재시작
./manage-rag.sh restart

# 4단계: 여전히 문제?
docker-compose down
docker-compose up -d
./manage-rag.sh status
```

### 일반적인 문제

**문제**: 웹UI에 접속 안 됨

```bash
./manage-rag.sh logs  # 에러 메시지 확인
curl http://localhost:3000  # 연결 테스트
```

**문제**: 모델 선택 불가

```bash
# Ollama 모델 확인
curl http://localhost:11434/api/tags

# 모델 다시 설치
docker exec rag-ollama ollama pull qwen2.5
```

**문제**: 메모리 부족

```bash
# 불필요한 컨테이너 정리
./manage-rag.sh clean

# 또는 수동 정리
docker system prune -a
```

---

## 💾 백업 & 복구

### 백업하기

```bash
# 한 줄 명령어
./manage-rag.sh backup

# 백업 위치: ./backups/backup_<날짜>.tar.gz
```

### 복구하기

```bash
# 이전 백업 목록 확인
ls -lh backups/

# 복구
./manage-rag.sh restore backups/backup_20260125_120000.tar.gz

# 서비스 재시작
./manage-rag.sh restart
```

---

## 🎯 배포 시나리오별 가이드

### 시나리오 1: 새로운 서버에 배포

```bash
# 1. 프로젝트 복제 (또는 파일 전송)
git clone <repo-url>
cd rag

# 2. 환경 설정
cp .env.example .env
# .env 파일 편집 (필요한 토큰 입력)

# 3. 시작
./start-rag.sh

# 4. 확인
./manage-rag.sh health-check
```

### 시나리오 2: 이미 배포된 서버 재시작

```bash
# 간단히 시작하면 됨
cd /Users/bagminjun/Desktop/rag
./start-rag.sh
```

### 시나리오 3: 소프트웨어 업데이트

```bash
# Git에서 최신 버전 가져오기
git pull origin main

# 또는 수동 파일 업데이트 후
docker-compose build sync-worker

# 시스템 재시작
./manage-rag.sh restart

# 확인
./manage-rag.sh status
```

### 시나리오 4: 긴급 종료

```bash
# 즉시 중지
docker-compose kill

# 또는 우아한 종료
./manage-rag.sh stop
```

---

## 📊 성능 모니터링

```bash
# 실시간 리소스 사용량
docker stats

# 디스크 사용량
df -h data/
du -sh data/*

# 컨테이너별 상세 정보
docker ps -a
docker inspect <container-name>
```

---

## 🔐 보안 점검

### 정기 점검 (월간)

```bash
# 1. 토큰/키 갱신
nano .env  # 필요한 토큰 업데이트

# 2. 포트 확인 (로컬호스트 전용)
docker-compose ps --format "table {{.Names}}\t{{.Ports}}"

# 3. 로그에서 의심 활동 확인
./manage-rag.sh logs | grep -i "error\|warn"

# 4. 백업 확인
ls -lh backups/
```

---

## 📞 도움말

### 스크립트 도움말

```bash
# 전체 명령어 확인
./manage-rag.sh help

# 또는
./manage-rag.sh
```

### 로그 기반 문제 해결

```bash
# 에러만 필터링
docker-compose logs | grep -i error

# 특정 시간대
docker-compose logs --since 1h

# 실시간 로그
docker-compose logs -f
```

---

## 🎓 고급 운영

### 자동 모니터링 설정 (Linux/macOS)

```bash
# Cron 작업으로 매일 헬스체크
# crontab -e 에서 추가:
0 9 * * * /Users/bagminjun/Desktop/rag/manage-rag.sh health-check >> /var/log/rag-healthcheck.log

# 주간 백업
0 2 * * 1 /Users/bagminjun/Desktop/rag/manage-rag.sh backup
```

### 원격 모니터링

```bash
# SSH를 통한 상태 확인
ssh user@server "cd /rag && ./manage-rag.sh status"

# 또는 모니터링 서비스 (Prometheus, etc.)
# docker-compose에 모니터링 추가 (선택사항)
```

---

## ✅ 최종 체크리스트

배포 완료 후 확인 사항:

- [ ] 모든 컨테이너 실행중
- [ ] 웹UI 접속 가능 (http://localhost:3000)
- [ ] 모델 선택 가능
- [ ] 문서 동기화 작동
- [ ] 로그에 에러 없음
- [ ] 백업 설정 완료
- [ ] 환경변수 안전 보관

---

**운영에 문제가 있으면 DEPLOYMENT_GUIDE.md 를 참고하세요!**
