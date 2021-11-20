

(이번학기에 배운 코드와 자연언어 처리 지식을 기반으로 구동 가능한 모델을 만들었습니다.)<br/>

###Check Mood Model
Depression model은 네이버 영화 평점 데이터를 이용하여 이용자가 하고 싶은 말을 입력하였을 때 당시의 기분 
혹은 현재 본인의 상태를 확인할 수 있도록 구성된 모델입니다.<br/>

####checkmood_model 구성

- raw_1 : 전처리 전 raw data
- modified : 전처리 후 data
- weights : 모델 weight이 저장되어 있는 파일
- cm_preprocessing.py : 전처리 과정
- cm_model.py : checkmood model

본 모델 실행을 위한 전처리 데이터는 cm_preprocessing.py를 통해 이미 준비가 되어 있기 때문에 cm_model.py만 실행하면
기분 상태를 확인할 수 있는 실행창을 열 수 있습니다. <br/>

####cm_preprocessing.py과 cm_model.py 
tokenizer = Okt <br/>
model = biLSTM (keras사용) <br/>
&nbsp;&nbsp; optimizer='rmsprop' <br/>
&nbsp;&nbsp; loss='categorical_crossentropy'


