import re

def preprocess_xml(input_file, source_lang_file):
    # 파일 열기
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()

    # <seg> 태그에서 텍스트 추출
    seg_pattern = r'<seg id="\d+">(.+?)<\/seg>'
    sentences = re.findall(seg_pattern, data)

    # 독일어 (소스)와 영어 (타겟) 분리 저장
    with open(source_lang_file, 'w', encoding='utf-8') as src_file:
        for sentence in sentences:
            # 여기서는 독일어(소스)만 처리한다고 가정
            src_file.write(sentence.strip() + '\n')

# 파일 경로 설정
input_file = '/home/user15/TT4/dataset9/dev/dev/newstest2013-src.en.sgm'  # XML 원본 파일
source_lang_file = '/home/user15/TT4/dataset9/dev/newstest2013.en'  # 독일어 (소스)

# 전처리 실행
preprocess_xml(input_file, source_lang_file)

# 파일 경로 설정
input_file = '/home/user15/TT4/dataset9/test/test-full/newstest2014-deen-ref.de.sgm'  # XML 원본 파일
source_lang_file = '/home/user15/TT4/dataset9/dev/newstest2013.en'  # 독일어 (소스)


# 전처리 실행
preprocess_xml(input_file, source_lang_file)