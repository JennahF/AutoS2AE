function A = neighbor2mat(pop_srt, N, mode)
    if (mode == 'ring')
        x = pop_srt;
        y = [pop_srt(2,:),pop_srt(1)];
        z = ones(1,N);
        A = sparse(x,y,z,N,N);
    elseif (mode == 'exp2')
        neighbor_num = ceil(log2(N));
        x=[];
        y=[];
        z=[];
        double_srt=[pop_srt,pop_srt];
        fixed_index=1:N;
        for i=1:neighbor_num
            x=[x,pop_srt];
            y=[y,double_srt(fixed_index+2^(i-1))];
            z=[z,ones(1,N)];
        end
        A=sparse(x,y,z,N,N);
    end
    end