# perl ./scripts/tokenizer/tokenizer.perl -l de -time -no-escape < ./newstest2014.de > ../dataset10/newstest2014.de
# perl ./scripts/tokenizer/tokenizer.perl -l en -time -no-escape < ./newstest2014.en > ../dataset10/newstest2014.en

# perl ./scripts/tokenizer/tokenizer.perl -l de -time -no-escape < ./newstest2013.de > ../dataset10/newstest2013.de
# perl ./scripts/tokenizer/tokenizer.perl -l en -time -no-escape < ./newstest2013.en > ../dataset10/newstest2013.en

# perl ./scripts/tokenizer/tokenizer.perl -l de -time -no-escape < ./train.de > ../dataset10/train.de
# perl ./scripts/tokenizer/tokenizer.perl -l en -time -no-escape < ./train.en > ../dataset10/train.en


# perl ./scripts/tokenizer/tokenizer.perl -l de -time -no-escape < /home/user15/TT4/dataset9/training/newscom/News-Commentary.de-en.de > ../dataset10/News-Commentary.de-en.de
# perl ./scripts/tokenizer/tokenizer.perl -l en -time -no-escape < /home/user15/TT4/dataset9/training/newscom/News-Commentary.de-en.en > ../dataset10/News-Commentary.de-en.en

cat /home/user15/TT4/dataset9/training/newscom/News-Commentary.de-en.de | ./cus_tokenizer.pl > /home/user15/TT4/dataset0/News-Commentary.de-en.de 

cat /home/user15/TT4/dataset9/training/newscom/News-Commentary.de-en.en | ./cus_tokenizer.pl > /home/user15/TT4/dataset0/News-Commentary.de-en.en


# cat ../dataset10/newstest2013.de | ./cus_tokenizer.pl > ../dataset11/newstest2013.de

# cat ../dataset10/newstest2013.en | ./cus_tokenizer.pl > ../dataset11/newstest2013.en


# cat ../dataset10/train.de | ./cus_tokenizer.pl > ../dataset11/train.de

# cat ../dataset10/train.en | ./cus_tokenizer.pl > ../dataset11/train.en




