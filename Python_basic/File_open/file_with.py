'''
# with 구문

with구문을 이용하면 따로 close()함수를 쓰지 않아도 with구문이 끝나면
자동으로 메모리 할당을 해제하기 때문에 간편하다.
'''

'''
1. input.txt 파일을 읽기모드로 불러오고, unicode 형태로 파일을 읽겠다.
   그리고 불러온 파일을 f라고 부른다.(변수 f에 담겠다.)
2. 데이터를 readlines()를 통해 한번에 다 읽고, 각 줄을 원소로 가지는 리스트를 반환 받는다.
   그걸 file2list에 넣는다.
3. 리스트의 원소와 원소의 인덱스에 따로 접근하는 enumerate를 사용하여 
   각 원소가 파일의 몇 번째 줄인지 확인하고자 한다.
   enumerate는 인덱스와 데이터. 이렇게 두 값에 접근하기 때문에 변수가 두개 필요하다.
   하나는 i(index)하나는 lines라고 이름 붙인다. 
4. 이대로 with구문을 끝내면 close()를 따로 안 써도 파일 객체에서 사용되었던 리소스가 해제된다.
'''

with open("input.txt", "r", encoding="UTF-8") as f:  # 1
    file2list = f.readlines()                        # 2
    for i, lines in enumerate(file2list):            # 3
        print("%d번째 줄: %s" %(i+1, lines), end='') # 4