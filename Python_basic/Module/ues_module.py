'''
다른 파일에서도 사용자 정의 모듈을 import해서 사용할 수 있다.
'''

# module이라는 이름의 모듈을 mm이라는 이름으로 불러온다.
# 모듈 파일 이름이 별로 안 길면 그냥 import만 해서 쓰면 된다.
import module as mm

print(mm.add(3, 7)) # module이라는 모듈에 있는 add함수를 사용한다.
