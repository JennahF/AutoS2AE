function [ train, test ] = readJester( data_dir )
%DataRead: read tuples from train and test files and transform them as a
%train and test sparse matrix
%   data_dir: gives the working directory of train and test files, in this
%   directory, there is a training file, named as train.txt and a testing
%   file, named as test.txt; each line in these two files consist of
%   user_id, item_id and rating; these fields are delimited by one tab

mat = xlsread(data_dir);

[numUsers, numItems]=size(mat);
index=mat==99;
mat(index)=0;
mat=mat(:,2:numItems);
numItems=numItems-1;
mat=sparse(mat);

[A,B,C]=find(mat);

termnum=length(mat);
portion=0.8;
trainIdx=floor(termnum*portion);

idx1 = C<=0;
idx2 = C>0;
C(idx1)=1;
C(idx2)=0;

randIdx = randperm(termnum);
A = A(randIdx);
B = B(randIdx);
C = C(randIdx);

C_train = {A(1:trainIdx),B(1:trainIdx),C(1:trainIdx)};
C_test = {A(trainIdx+1:termnum),B(trainIdx+1:termnum),C(trainIdx+1:termnum)};

train = sparse(C_train{1}, C_train{2}, C_train{3}, numUsers, numItems);
test = sparse(C_test{1}, C_test{2}, C_test{3}, numUsers, numItems);
end

