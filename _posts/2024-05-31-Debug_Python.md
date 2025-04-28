---
title: Python 오류모음
description: For Python Debug
date: 2025-04-23 11:33:00
categories: [Debug, Python]
pin: true
math: true
mermaid: true
---
># Python 오류 정리
Python에 관련한 오류 정리

<details>
<summary>시각화 한글 인식 오류</summary>

```python
"""
시각화 출력 시 ㅁ로 뜨는 현상
in Mac
"""
# 현재 가지고 있는 폰트 리스트 확인.
import matplotlib.font_manager as fm
font_list = [font.name for font in fm.fontManager.ttflist]
font_list

>>> ['현재 가지고 있는 폰트들 리스트가 뜸']

# 찾을 때 원하는 폰트 있는지 확인
# 여기서 만약 없으면 Mac 기준 Library/Fonts 경로 폴더에 다운로드해서 넣어야 함.
if 'AppleGothic' in font_list:
    print('yes')
else:
    print('No')

>>> yes

# 특정 폰트를 지정하는 코드
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False # 음의 값을 다룬다면 True로

# 이후에 간단히 확인하기 위해 출력해보기
x = [1,2,3]
y = [3,5,5]

plt.plot(x,y, marker = 'o')
plt.title('막대그래프')
plt.xlabel('값1')
plt.ylabel('값2')
plt.grid(True)
plt.show()
```

</details>