# en-de
# wget https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz # Europarl v7
# wget https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz # Common Crawl corpus
#wget https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz # News Commentary v9

# # Common Crawl 데이터
# tar -xzvf /home/user15/TT4/dataset9/training/training-parallel-commoncrawl.tgz -C /home/user15/TT4/dataset9/cc/
# # Europarl 데이터
# tar -xzvf /home/user15/TT4/dataset9/training/training-parallel-europarl-v7.tgz -C /home/user15/TT4/dataset9/eu/
# # News Commentary 데이터
# tar -xzvf /home/user15/TT4/dataset9/training/training-parallel-nc-v9.tgz -C /home/user15/TT4/dataset9/training/newscom


# cat /home/user15/TT4/dataset9/training/cc/commoncrawl.de-en.de \
#     /home/user15/TT4/dataset9/training/eu/europarl-v7.de-en.de \
#     /home/user15/TT4/dataset9/training/newscom/news-commentary-v9.de-en.de > /home/user15/TT4/dataset9/training/train.de

# cat /home/user15/TT4/dataset9/training/cc/commoncrawl.de-en.en \
#     /home/user15/TT4/dataset9/training/eu/europarl-v7.de-en.en \
#     /home/user15/TT4/dataset9/training/newscom/news-commentary-v9.de-en.en > /home/user15/TT4/dataset9/training/train.en



# cat /home/user15/TT4/dataset10/train.de \
#     /home/user15/TT4/dataset11/News-Commentary.de-en.de  > /home/user15/TT4/dataset11/train.de

# cat /home/user15/TT4/dataset10/train.en \
#     /home/user15/TT4/dataset11/News-Commentary.de-en.en  > /home/user15/TT4/dataset11/train.en

# cat /home/user15/TT4/dataset0/commoncrawl.de-en.en \
#     /home/user15/TT4/dataset0/europarl-v7.de-en.en \
#     /home/user15/TT4/dataset0/news-commentary-v9.de-en.en > /home/user15/TT4/dataset0/train.en

# wc -l /home/user15/TT4/dataset0/train.de
# wc -l /home/user15/TT4/dataset0/train.en




cat /home/user15/TT4/dataset1/commoncrawl.de-en.de \
    /home/user15/TT4/dataset1/europarl-v7.de-en.de \
    /home/user15/TT4/dataset1/News-Commentary.de-en.de \
    > /home/user15/TT4/dataset1/train.de

cat /home/user15/TT4/dataset1/commoncrawl.de-en.en \
    /home/user15/TT4/dataset1/europarl-v7.de-en.en \
    /home/user15/TT4/dataset1/News-Commentary.de-en.en \
    > /home/user15/TT4/dataset1/train.en


wc -l  /home/user15/TT4/dataset1/train.de
wc -l  /home/user15/TT4/dataset1/train.en