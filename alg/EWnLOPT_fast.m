function [P,Q, output_metric, a_l, b_l, l_l, v_l, r_l, n_l] = EWnLOPT_fast(R_t, R_v, fun, log_file, varargin)
[M, N] = size(R_t);


[alpha, test, beta, topk, max_iter, lr, neighber_mat] = process_options(varargin, 'alpha', 10, 'test', [],'beta', 250, 'topk', 200, 'max_iter', 10, 'lr', 0.01, 'neighbor', []);

max_iter
fprintf('alpha=%f, beta=%f)\n', alpha, beta);
    
P = [];
Q = [];
fprintf('Start calculating')
f = fopen(log_file, 'a');


% if(~isempty(neighber_mat))
%     fprintf(f, 'Sparsity:%f\n', sum(sum(neighber_mat))/(N*N))
% end

a_l = [];
b_l = [];
l_l = [];
v_l = [];
r_l = [];
n_l = [];
temp = 0;

p = 1
v_idx = sum(R_v)'; %N*1
v_idx = v_idx~=0;
prob = rand(N,1)<p;
v_idx = v_idx&prob;

a_sr = {0 0};
b_sr = {0 0};
t = 0;

for i=1:max_iter
    tic;[P, Q, parBalpha, parBbeta] = ease_weight(R_t, v_idx, alpha, beta, fun, neighber_mat);toc;
    
    a_l = [a_l,alpha];
    b_l = [b_l, beta];

    fprintf('Start testing');
    fprintf(f, '\niter %d', i);
    fprintf(f, '\nalpha=%f, beta=%f', alpha, beta);
    if (~isempty(test))
        tic;metric = evaluate_item(R_t+R_v, test, P, Q.', 200, 200);toc;
        
        if i == 1
            output_metric = {metric, alpha, beta};
        elseif metric.item_recall(1,10) >= output_metric{1}.item_recall(1,10)
            output_metric = {metric, alpha, beta};
        end
        
        if isexplict(test)
            fprintf(f,'\nrecall@10,20,50=%.5f,%.5f,%.5f', metric.item_recall_like(1,10), metric.item_recall_like(1,20), metric.item_recall_like(1,50));
            fprintf(f,'\nndcg@10,20,50=%.5f,%.5f,%.5f', metric.item_ndcg_like(1,10), metric.item_ndcg_like(1,20), metric.item_ndcg_like(1,50));
        else
            fprintf(f,'\nrecall@10,20,50=%.5f,%.5f,%.5f', metric.item_recall(1,10), metric.item_recall(1,20), metric.item_recall(1,50));
            fprintf(f,'\nndcg@10,20,50=%.5f,%.5f,%.5f', metric.item_ndcg(1,10), metric.item_ndcg(1,20), metric.item_ndcg(1,50));
        end
    end

    if alpha <= 0
        break
    end

    loss = loss_func(P, Q, alpha, beta);
    l_l = [l_l, loss];
    % valid_l = valid_loss(P, Q, alpha, beta)
    % v_l = [v_l, valid_l]


    % tic;[alpha, beta] = Update_para(R_v, Q, alpha, beta, parBalpha, parBbeta, lr);toc;
    tic;[alpha, beta, a_sr, b_sr] = Update_para_Adam(R_v, Q, alpha, beta, parBalpha, parBbeta, lr, a_sr, b_sr, t);toc;

    

    fprintf(f,'\nloss=%.5f\n', loss);
    r_l = [r_l,metric.item_recall(1,10)];
    n_l = [n_l, metric.item_ndcg(1,10)];
end
fclose(f);

end

function loss = loss_func(P, Q, alpha, beta)
    w = 1+alpha*P;
    pred = P-P*Q;
    pred = pred.^2;
    l1 = sum(sum(w.*pred));
    l2 = beta*sum(sum(Q.^2));
    loss = full(l1+l2);
end

function [alpha, beta] = Update_para_SGD(R, B, alpha, beta, parBalpha, parBbeta, lr)

    % error = R.*(R-R*B);
    error = R-R*B;
    Ralpha = R*parBalpha;
    Rbeta = R*parBbeta;

    alpha_dir = 2*sum(sum(error.*Ralpha));
    beta_dir = 2*sum(sum(error.*Rbeta));
    alpha = alpha+lr*alpha_dir;
    beta = beta+lr*beta_dir;
    if alpha < 0
        alpha = 0;
    end
    if beta < 0
        beta = 0;
    end
end

function [alpha, beta, a_sr, b_sr] = Update_para_Adam(R, B, alpha, beta, parBalpha, parBbeta, lr, a_sr, b_sr, t)

    h1 = 0.9;
    h2 = 0.999;
    epsilon = 0.01;
    t = t+1;

    [s_a, r_a] = a_sr{:};
    [s_b, r_b] = b_sr{:};

    error = R-R*B;
    Ralpha = R*parBalpha;
    Rbeta = R*parBbeta;
    alpha_dir = -2*sum(sum(error.*Ralpha));
    beta_dir = -2*sum(sum(error.*Rbeta));

    s_a = h1*s_a+(1-h1)*alpha_dir;
    s_b = h1*s_b+(1-h1)*beta_dir;

    r_a = h1*r_a+(1-h1)*alpha_dir*alpha_dir;
    r_b = h1*r_b+(1-h1)*beta_dir*beta_dir;

    c = sqrt(1-h2^t)/sqrt(1-h1^t);

    alpha = alpha-lr*c*s_a/(sqrt(r_a)+epsilon);
    beta = beta-lr*c*s_b/(sqrt(r_b)+epsilon);

    a_sr = {s_a r_a};
    b_sr = {s_b r_b};

end

function [neighbers] = KNN_neighbers(R)
    [M,N] = size(R);
    R=R*1./sqrt(sum(R.^2,1)); % norm
    k = floor(0.1*N);
    similarity = R'*R;
    similarity(1:N+1:end) = 0;
    [s1, ind] = maxk(similarity, k, 1);
    col = ones(size(ind));
    for i = 2:N
        col(:,i)=i;
    end
    col = col(:);
    ind = ind(:);
    neighbers = sparse(ind, col, 1, N, N);
    % s2 = mink(s1, 1, 1);
    % neighbers = similarity>s2;
end

function [P, Q, parBalpha, parBbeta] = ease_weight(R, v_idx, alpha, beta, fun, neighber_mat)

    iter_times = 10000;
    epsilon = 1e-3;
    neighbers = sparse(R'*R);
    RtR = neighbers;
    [M,N] = size(R);

    if (~isempty(neighber_mat))
        neighbers = neighber_mat;
    else
        % neighbers = neighbers>0;
        neighbers = KNN_neighbers(R);
        % neighbers = ones(N,N);
    end
    neighbers(1:N+1:end) = 0;
    
    sparsity = sum(sum(neighbers))/(N*N)

    B = spalloc(N,N,nnz(neighbers));
    fcn = @plus;
    parBbeta = spalloc(N,N,nnz(neighbers));
    parBalpha = spalloc(N,N,nnz(neighbers));

    fprintf("loop start!\n")
    parfor j=1:N
        if (j == floor(N/4) | j == floor(N/2) | j == floor(N*3/4) | j == N)
            fprintf("%d\n", j);
        end

        non_zero_entries = neighbers(:,j)~=0;
        X = R(:, non_zero_entries);
        Xt = X';
        [temp,X_n] = size(X);
        xj = R(:,j);
        Xj = Xt(:,xj~=0);
        XtX = RtR(:,non_zero_entries)';
        XtX = XtX(:,non_zero_entries)'; % symmetric matrix
        E = speye(X_n);

        XjtXj = Xj*Xj';

        S = XtX+alpha*XjtXj+beta*E;
        Xtxj = RtR(:,j);
        Xtxj = Xtxj(non_zero_entries,:);

        b = (alpha+1)*Xtxj;

        idx = find(non_zero_entries);
        b0 = fun(S, b);
        mat0 = sparse(idx,ones(1,length(idx))*j, b0,N,N);
        B = fcn(B, mat0);

    %     % partial

    %     % non_zero_entries = (neighbers(:,j)~=0) & v_idx;
    %     % X = R(:, non_zero_entries);
    %     % Xt = X';
    %     % [temp,X_n] = size(X);
    %     % xj = R(:,j);
    %     % Xj = Xt(:,xj~=0);
    %     % XtX = RtR(:,non_zero_entries)';
    %     % XtX = XtX(:,non_zero_entries)'; % symmetric matrix
    %     % E = speye(X_n);

    %     % XjtXj = Xj*Xj';

    %     % S = XtX+alpha*XjtXj+beta*E;
    %     % Xtxj = RtR(:,j);
    %     % Xtxj = Xtxj(non_zero_entries,:);

    %     % b = (alpha+1)*Xtxj;

        mat = sparse(idx,ones(1,length(idx))*j,fun(-S,b0),N,N);
        parBbeta = fcn(parBbeta, mat);


        b1 = XjtXj*b0;
        mat1 = sparse(idx,ones(1,length(idx))*j,fun(-S,b1),N,N);
        parBalpha = fcn(parBalpha, mat1);
        parBalpha = fcn(parBalpha, mat0/(alpha+1));
    end
    % % matlabpool close;
    % B = sparse(rows, cols, values, N, N);
    B(1:N+1:end) = 0;
    parBbeta(1:N+1:end) = 0;
    parBalpha(1:N+1:end) = 0;
    P = R;
    Q = B;

end
