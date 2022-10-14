function [intersectVectors, ind_a, ind_b] = intersectrows_fast_v1(a,b)
% 
% %// Calculate equivalent one-column versions of input arrays
% 
% mult = [10^ceil(log10( 1+max( [a(:);b(:)] ))).^(size(a,2)-1:-1:0)]'; %//'
% 
% acol1 = a*mult;
% 
% bcol1 = b*mult;
% 
% %// Use intersect without 'rows' option for a good speedup
% 
% [~, ind_a, ind_b] = intersect(acol1,bcol1);
% 
% intersectVectors = a(ind_a,:);


%// Calculate equivalent one-column versions of input arrays

mult = [10^ceil(log10( 1+max( [a(:);b(:)] ))).^(size(a,2)-1:-1:0)]'; %//'

acol1 = a*mult;

bcol1 = b*mult;

%// Use ismember to get indices of the common elements

[match_a,idx_b] = ismember(acol1,bcol1);

%// Now, with ismember, duplicate items are not taken care of automatically as

%// are done with intersect. So, we need to find the duplicate items and

%// remove those from the outputs of ismember

[~,a_sorted_ind] = sort(acol1);

a_rm_ind =a_sorted_ind([false;diff(sort(acol1))==0]); %//indices to be removed

match_a(a_rm_ind)=0;

intersectVectors = a(match_a,:);

ind_a = find(match_a);

ind_b = idx_b(match_a);

return;
