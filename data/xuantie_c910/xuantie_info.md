# Xuantie C910 (玄铁 C910) RISC-V 프로세서

## 기본 정보
- 정식 명칭: Xuantie C910 (玄铁 C910), 이전 명칭: XuanTie 910
- 개발사: T-Head Semiconductor (平头哥半导体), Alibaba Group 산하 반도체 자회사
- ISA: RISC-V (RV64GCV)
- 최초 공개: 2019년 7월 (Alibaba Cloud Summit)
- 오픈소스 공개: 2021년 10월 (GitHub)

## 아키텍처 특징
- 64비트 RISC-V 코어
- 12단계 파이프라인 (슈퍼스칼라, 최대 3-issue)
- 비순차 실행(Out-of-Order Execution) 지원
- 벡터 확장(V Extension) 0.7.1 지원 (128비트 벡터 레지스터)
- MMU: Sv39/Sv48 가상 메모리
- 캐시: L1 I-Cache 64KB, L1 D-Cache 64KB, L2 Cache 최대 2MB
- 멀티코어: 최대 16코어 클러스터 구성 가능
- 최대 클럭: 2.5GHz (12nm 공정 기준)

## 성능
- CoreMark 스코어: 7.1 CoreMark/MHz
- SPECint2006: ARM Cortex-A73 대비 약 40% 높은 IPC
- AI 추론: 벡터 확장으로 INT8/FP16 추론 가속
- 리눅스 부팅: 완전한 Linux 커널 지원

## 주요 응용 분야
- 엣지 AI 컴퓨팅
- 네트워크 장비 (5G 기지국, 라우터)
- 자율주행 보조 시스템
- 클라우드 컴퓨팅 서버
- IoT 게이트웨이

## RISC-V 생태계에서의 위치
- 공개 당시 가장 고성능 RISC-V 프로세서 코어
- GitHub에서 오픈소스로 공개 (Apache 2.0 라이선스)
- 중국 RISC-V 생태계의 핵심 IP
- Alibaba의 Wujian SoC 플랫폼에서 활용

## T-Head(平头哥) 소개
- 2018년 설립 (Alibaba의 중천미전자 + 다모아카데미 반도체 부문 합병)
- 명칭 유래: 꿀오소리(Honey Badger, 平头哥)
- 주요 제품: Xuantie 시리즈 (C906, C908, C910, C920)
- C920: C910의 후속 모델, RVV 1.0 지원, 최대 4GHz

## 기술적 의의
- RISC-V ISA가 고성능 컴퓨팅 영역에서도 경쟁력 있음을 입증
- x86/ARM 독점 구도에 대한 대안 제시
- 오픈소스 공개로 RISC-V 생태계 성장에 기여
- 중국의 반도체 자립(국산화) 전략의 핵심 자산
