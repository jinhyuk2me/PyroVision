# EO-IR 화재 감지 융합 시스템 로드맵

## 개요

EO(RGB) 카메라와 IR(열화상) 카메라의 데이터를 융합하여 안정적인 화재 감지 시스템을 구축합니다.

### 현재 상태
- **IR**: 온도 기반 hotspot 감지 + 캡처/재생/목업 파이프라인 구현 완료
- **EO**: YOLOv8 기반 화염 감지 + HDMI 디스플레이/송출/재현 테스트 지원
- **기능 현황**  
  1. Phase 1 (IR 게이트키퍼) 구현됨  
  2. 캡처 → 재생 → 동기화 테스트 루틴 구축  
  3. mock/video 입력으로 CI/개발 테스트 가능

### 목표
- 오검지(False Positive) 최소화
- 미검지(False Negative) 최소화  
- 신뢰도 기반 판정 시스템 구축

---

## Phase 1: 기본 융합 (IR 게이트키퍼) ✅ 완료

### 개념
IR hotspot 감지를 필수 조건으로 설정하여 EO 단독 오검지를 방지합니다.

### 판정 로직
```
IR hotspot 감지?
    ├─ NO  → 최종: NOT FIRE (EO 결과 무시)
    │
    └─ YES → EO fire bbox와 겹침?
                ├─ YES → FIRE (HIGH confidence: 0.95)
                └─ NO  → FIRE (MEDIUM confidence: 0.7)
```

### 구현 범위
1. `detect_fire()` 반환값 확장 (hotspots 포함)
2. IR-EO 좌표 매핑 (선형 스케일링)
3. 기본 융합 로직 (`core/fire_fusion.py`)
4. 결과 전송 (`sender.py` 확장)

### 파라미터
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| IR_MIN_TEMP | 80°C | IR hotspot 최소 온도 |
| MATCH_CONFIDENCE | 0.95 | IR+EO 매칭 시 신뢰도 |
| IR_ONLY_CONFIDENCE | 0.7 | IR만 감지 시 신뢰도 |

### 현 시점 한계
- EO 단독 감지 무시 (초기 화재 미검지 가능성)
- 좌표/시간 정합 미흡 → 정확한 융합 한계
- 신뢰도/상태 관리 부재

---

## Phase 2: 좌표 캘리브레이션 (우선 진행)

### 개념
키보드 기반 실시간 좌표 오프셋 조정으로 IR-EO 정렬 정확도 향상

### 구현 범위
1. `core/coord_mapper.py` 확장
   - 오프셋 X/Y 파라미터
   - 스케일 조정
   - 회전 보정 (옵션)

2. 키보드 컨트롤 추가
   ```
   [i/k] Y축 오프셋 조정
   [j/l] X축 오프셋 조정
   [+/-] 스케일 조정
   [c]   캘리브레이션 저장
   ```

3. Config 확장
   ```yaml
   COORD_MAP:
     OFFSET_X: 0
     OFFSET_Y: 0
     SCALE: 4.5
   ```

### 기대 효과
- 카메라 설치 위치에 따른 정확한 매핑
- 런타임 조정 가능

---

## Phase 3: 시간 동기화 (Phase2와 병행/후속)

### 개념
IR (9fps)과 EO (30fps) 간 프레임 타임스탬프 동기화

### 구현 범위
1. 타임스탬프 기반 프레임 매칭
2. 최근접 프레임 선택 로직
3. 허용 시간차 파라미터 (예: 100ms)

### 로직
```python
def find_matching_frame(ir_timestamp, eo_frames, max_diff_ms=100):
    for eo_frame in eo_frames:
        if abs(ir_timestamp - eo_frame.timestamp) < max_diff_ms:
            return eo_frame
    return None
```

---

## Phase 4-5: 신뢰도 + 상태 머신

### 개념
다양한 요소를 고려한 동적 신뢰도 계산

### 신뢰도 요소
| 요소 | 가중치 | 설명 |
|------|--------|------|
| IR 온도 | 0.3 | 높을수록 신뢰도 증가 |
| EO confidence | 0.3 | YOLOv8 검출 신뢰도 |
| 영역 겹침 (IoU) | 0.2 | 겹침 비율 |
| 지속 시간 | 0.2 | 연속 감지 프레임 수 |

### 신뢰도 계산
```python
def calculate_confidence(ir_temp, eo_conf, iou, duration):
    temp_score = min(1.0, (ir_temp - 80) / 100)  # 80~180°C → 0~1
    
    confidence = (
        0.3 * temp_score +
        0.3 * eo_conf +
        0.2 * iou +
        0.2 * min(1.0, duration / 30)  # 30프레임 이상 → 1.0
    )
    return confidence
```

---

## Phase 6: 다중 화점 추적 (선택적 고도화)

### 개념
여러 화점을 개별적으로 추적 및 관리

### 구현 범위
1. 화점별 고유 ID 부여
2. 프레임 간 화점 매칭 (헝가리안 알고리즘)
3. 화점별 상태 머신 관리
4. 화점 병합/분리 처리

### 데이터 구조
```python
class FirePoint:
    id: int
    ir_positions: List[Tuple[int, int]]  # 이력
    eo_bboxes: List[Tuple]  # 이력
    temperatures: List[float]  # 이력
    state: str
    confidence: float
    first_seen: float
    last_seen: float
```

---

## Phase 7: 적응형 임계값 (장기 과제)

### 개념
환경 조건에 따른 동적 임계값 조정

### 고려 요소
- 주변 평균 온도 (배경 온도)
- 시간대 (주간/야간)
- 계절/날씨
- 과거 오검지 패턴

### 구현 예시
```python
def adaptive_threshold(background_temp, time_of_day):
    base_threshold = 80  # 기본 80°C
    
    # 배경 온도 보정
    if background_temp > 35:
        base_threshold += (background_temp - 35) * 0.5
    
    # 야간 보정 (열원이 더 두드러짐)
    if 22 <= time_of_day or time_of_day <= 6:
        base_threshold -= 5
    
    return base_threshold
```

---

## 구현 우선순위 (업데이트)

| 순서 | 단계 | 핵심 내용 |
|------|------|-----------|
| 1 | Phase 2 | 좌표 캘리브레이션 UI + Config |
| 2 | Phase 3 | 융합 단계 동기화 + 버퍼 매칭 |
| 3 | Phase 4-5 | 신뢰도 계산 + 상태 머신 |
| 4 | Phase 6 | 다중 화점 추적 |
| 5 | Phase 7 | 적응형 임계값 |

---

## 테스트 전략

1. **캡처 → 재생 회귀**: `capture.py`로 표준 IR/RGB 세션을 저장하고, 각 Phase 구현 후 동일 데이터를 재생해 결과 비교  
2. **Mock 입력**: `INPUT.MODE=mock`으로 CI/개발 환경에서 반복 가능한 테스트 수행  
3. **현장 검증**: HDMI 디스플레이 + TCP 송출을 동시에 켜고, Phase별 변경 사항을 실제 장비에서 확인  
4. **SYNC 검증**: `SYNC.ENABLED`를 켜고 IR/RGB 시간차 허용치를 조정해 동기화 실패/성공 시나리오 검증  

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2024-11-28 | 0.1 | 초안 작성 |
| 2025-02-XX | 0.2 | Phase 정리/캡처·재생 파이프라인 반영, 우선순위 재정립 |
| 2025-02-XX | 0.2 | Phase 정리/캡처·재생 파이프라인 반영, 우선순위 재정립 |
