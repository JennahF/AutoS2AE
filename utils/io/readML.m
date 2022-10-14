function [ data ] = readML( data_dir )
%DataRead: read tuples from train and test files and transform them as a
%train and test sparse matrix
%   data_dir: gives the working directory of train and test files, in this
%   directory, there is a training file, named as train.txt and a testing
%   file, named as test.txt; each line in these two files consist of
%   user_id, item_id and rating; these fields are delimited by one tab
f = fopen(data_dir);
% data_dir
% if data_dir == 'test/dataset/ml-20m.csv'
    % C = textscan(f,'%f,%f,%f,%d','HeaderLines',1);
% else
    C = textscan(f,'%f::%f::%f::%d');
% end
fclose(f);

numUsers = max(C{1});
numItems = max(C{2});
minU = min(C{1})
minI = min(C{2})
data = sparse(C{1}, C{2}, C{3}, numUsers, numItems);
end
