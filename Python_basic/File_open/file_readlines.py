'''
# readlines()

readlines()함수는 모든 줄을 한번에 쫙 읽어내는 함수이다.
파일 객체로부터 모든 내용을 읽어내 리스트에 담는다.
'''
f = open('input.txt', 'r', encoding='UTF-8')

file2list = f.readlines() # file2list에 readlines()수행 결과를 담는다.
# print(file2list)
# -> ['세상아\n', '난\n', '칠성사이다\n', '스프라이트\n', '다 좋아']
# 각각의 줄을 하나의 원소로 가지는 하나의 리스트를 반환한다.
# \n은 줄바꿈을 의미한다.
'''
# 위와 같은 리스트의 각 원소에 인덱스를 달아 "%d번째 줄: %s" 형태로 만들어보겠다.
1. enumerate를 사용하여 해당 리스트의 각 인덱스와 데이터에 따로따로 접근할 수 있도록 한다.
   인덱스와 원소를 구분하여 접근하기 때문에 각각의 변수가 필요하다.
   인덱스를 담을 변수i, 원소를 담을 변수 lines를 만들어 각각을 담는다.
2. i는 index이기 때문에 0부터 센다. 그래서 '몇 번째 줄'이라고 출력하고 싶으면 +1을 해야 한다.
'''
for i, lines in enumerate(file2list):              # 1
    print("%d번째 줄: %s" %(i + 1, lines), end='') # 2
f.close()