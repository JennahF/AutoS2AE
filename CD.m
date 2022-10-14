function [xi] = CD(A,x,y,i)
    Ai = A(:,i);
    % A(:,i) = 0;
    r = y-A*x;
    xi = x(i)+(Ai'*r)/(Ai'*Ai);
end