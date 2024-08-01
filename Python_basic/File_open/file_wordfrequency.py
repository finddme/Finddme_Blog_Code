'''
open()과 read()를 사용하여
파일을 불러와 해당 파일에 포함된 문자의 빈도수를 확인하는 함수를 만들어보자
'''

'''
1. 특정 파일의 이름을 넣으면 그 파일을 불러와 처리할 수 있는 함수를 만들어보자
2. 함수에 파일 이름이 입력되면 그 파일을 읽기모드로 불러와 unicode형식으로 읽고
   그 파일을 f라고 부른다.
3. 각 문자의 빈도수를 추출할 것이기 때문에 '문자:빈도수' 형식을 가질 수 있는
   딕셔너리 자료형을 하나 만들자
4. read()함수를 이용해 불러온 파일을 읽자. 읽어온 파일은 words변수에 넣는다.
5. 이제 파일 내에 있는 문자들을 하나씩 돌면서 빈도수를 확인한다.
6. 해당 문자가 딕셔너리 자료형 안에 있으면, 
   당시 변수 i가 담고 있는 문자가 key로 존재하는 원소의 value값에 1을 더하고
7. 그렇지 않으면 i가 담고 있는 문자가 key로 존재하고 value는 1인 원소를 생성
8. 이렇게 채워진 dict1을 반환한다.
'''
def frequency(file):                               # 1
    with open(file, "r", encoding="UTF-8") as f: # 2
        dict1 = {}                                 # 3
        words = f.read()                           # 4
        for i in words:                            # 5
            if i in dict1:                         # 6
                dict1[i] += 1
            else:                                  # 7
                dict1[i] = 1
    return dict1                                   # 8

word_frequency = frequency("input.txt")

'''
빈도수를 내림차순으로 정렬하여 가장 높은 빈도수를 지닌 문자를 확인해보자
- items()함수를 이용하면 딕셔너리의 key와 value 쌍으로 이루어진 리스트가 반환된다.
  key와 value쌍으로 이루어진 원소는 튜플형태이다.
- sorted()함수는 key, reverse 매개변수를 갖느다.
  key는 정렬 기준을 설정해주는 매개변수이고
  reverse는 False일 때 오름차순, True일 때 내림차순 정렬을 설정해준다.
- key를 lambda와 함께 사용하였는데, a를 매개변수로 받았을 때 a의 인덱스 1을 key값으로 하겠다는 것이다.
- 그리고 인덱스 1을 내림차순으로 정렬하겠다고 reverse=True를 해준다.
- 그 결과를 word_frequency에 담는다.

'''
word_frequency = sorted(word_frequency.items(), key = lambda a : a[1], reverse=True)

# print(word_frequency)
# -> [('\n', 4), ('아', 2), ('이', 2), ('다', 2), ('세', 1), ('상', 1), ('난', 1), ('칠', 1),
# ('성', 1), ('사', 1), ('스', 1), ('프', 1), ('라', 1), ('트', 1), (' ', 1), ('좋', 1)]
# 그 결과가 이렇게 나온다. 이제 위 리스트의 각 원소에 접근해서 "%d번 출현: [%s]"의 형태로 뽑아보겠다.

'''
1. 리스트 내에 튜플로 이루어진 원소는 두 원소로 이루어져 있기 때문에
   word_frequency에 한번 방문할 때 두 값을 담을 두 변수가 필요하다.
   튜플의 첫 번째 원소는 data변수에 담고, 두 번째 원소는 count에 담는다
2. 문자 중에서 줄바꿈이나 공백은 무시한다.
'''
for data, count in word_frequency:   # 1
    if data == "\n" or data == " ":  # 2
        continue
    print("%d번 출현: [%c]"%(count, data))