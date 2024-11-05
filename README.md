<h2> How To Use </h2>

1. 파이썬 가상 환경에서 requirements.txt 다운. (3.10 버전 권장)
```python
pip install -r requirements.txt
```

2. detect.py 실행
   - run 함수가 메인 동작 기능입니다. run 함수의 매개변수 값을 변경하면 source(ex. img.jpg, webCam)와 가중치(best.pt), 매개변수(threshold)값을 조절할 수 있습니다.
   - 탐지된 이미지는 runs 파일에 저장되며 서버에는 하드코딩된 위도와 경도가 보내집니다. 서버의 응답을 해주는 코드가 없어 정상적으로 보낸더라도 500이 뜹니다.
   - 포트홀 탐지 시 box 모양의 오브젝트 추가하는 코드를 제거하여 원본 데이터가 저장됩니다. box 모양이 있고 싶다면 '''로 된 주석 블럭을 지우면 됩니다.
  
리팩터링과 수정이 매우 필요