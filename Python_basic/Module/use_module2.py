'''
from구문을 통해서도 모듈을 사용할 수 있다.
'''

from module import add

# from으로 해당 모듈의 특정 함수를 불러왔으면 따로 module.add라고 쓰지 않아도 된다.
print(add(3, 8))