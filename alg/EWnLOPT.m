function [P, Q, output_metric, a_l, b_l, l_l, r_l, n_l] = EWnLOPT(R_t, R_v, fun, varargin)
[M, N] = size(R_t);


[alpha, test, beta, topk, max_iter, lr] = process_options(varargin, 'alpha', 10, 'test', [],'beta', 250, 'topk', 200, 'max_iter', 10, 'lr', 0.01);

max_iter
fprintf('alpha=%f, beta=%f)\n', alpha, beta);
    
P = [];
Q = [];
fprintf('Start calculating')
f = fopen("gowalla.txt", 'w+');
a_l = [];
b_l = [];
l_l = [];
r_l = [];
n_l = [];
temp = 0;

for i=1:max_iter
    tic;[P, Q, parBalpha, parBbeta] = ease_weight(R_t, alpha, beta, fun);toc;
    tic;[alpha, beta] = Update_para(R_v, Q, alpha, beta, parBalpha, parBbeta, lr);toc;
    loss = loss_func(P, Q, alpha, beta);
    a_l = [a_l,alpha];
    b_l = [b_l, beta];
    l_l = [l_l, loss];
    
    if alpha < 10
        temp = 1;
    end
    
    fprintf('Start testing');
    fprintf(f, '\niter %d', i);
    fprintf(f, '\nalpha=%f, beta=%f', alpha, beta);
    if (~isempty(test))
        tic;metric = evaluate_item(R_t, test, P, Q, 200, 200);toc;
        
        if i == 1
            output_metric = {metric, alpha, beta};
        elseif metric.item_recall(1,50) >= output_metric{1}.item_recall(1,50)
            output_metric = {metric, alpha, beta};
        end
        
        if isexplict(test)
            fprintf(f,'\nrecall@10,20,50=%.3f,%.3f,%.3f', metric.item_recall_like(1,10), metric.item_recall_like(1,20), metric.item_recall_like(1,50));
            fprintf(f,'\nndcg@10,20,50=%.3f,%.3f,%.3f', metric.item_ndcg_like(1,10), metric.item_ndcg_like(1,20), metric.item_ndcg_like(1,50));
        else
            fprintf(f,'\nrecall@10,20,50=%.3f,%.3f,%.3f', metric.item_recall(1,10), metric.item_recall(1,20), metric.item_recall(1,50));
            fprintf(f,'\nndcg@10,20,50=%.3f,%.3f,%.3f', metric.item_ndcg(1,10), metric.item_ndcg(1,20), metric.item_ndcg(1,50));
        end
    end
    fprintf(f,'loss=%.3f\n', loss);
    r_l = [r_l,metric.item_recall(1,10)];
    n_l = [n_l, metric.item_ndcg(1,10)];
    if alpha == 0
        break
    end
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

function [alpha, beta] = Update_para(R, B, alpha, beta, parBalpha, parBbeta, lr)

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

function [P, Q, parBalpha, parBbeta] = ease_weight(R, alpha, beta, fun)

    iter_times = 10000;
    epsilon = 1e-3;
    neighbers = sparse(R'*R);
    RtR = neighbers;
    neighbers = neighbers>0;
%     neighbers(1:N+1:end) = 0;
    
    [M,N] = size(R);
    sparsity = sum(sum(R))/(M*N)

    B = spalloc(N,N,nnz(neighbers));
    fcn = @plus;
    parBbeta = spalloc(N,N,nnz(neighbers));
    parBalpha = spalloc(N,N,nnz(neighbers));

    fprintf("loop start!\n")
    parfor j=1:N
        if (j == floor(N/4) | j == floor(N/2) | j == floor(N*3/4) | j == N)
            fprintf("%d\n", j);
        end

        % non_zero_entries = neighbers(:,j)~=0;
        % X = (R(:, non_zero_entries))';
        % xj = R(:,j);
        % Xj = X(:,xj~=0);
        % [X_n, temp] = size(X);
        % E = speye(X_n);
        % S_inv = X*X'+alpha*(Xj*Xj')+beta*E;
        % B(non_zero_entries,j) = S_inv\((alpha+1)*(X*xj));


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

        % S_inv = inv(S);
        % b = (alpha+1)*Xtxj;
        % Sinvb = S_inv*b;

        % idx = find(non_zero_entries);
        % mat0 = sparse(idx,ones(1,length(idx))*j,Sinvb,N,N);
        % B = fcn(B, mat0);
        % mat = sparse(idx,ones(1,length(idx))*j,-S_inv*Sinvb,N,N);
        % parBbeta = fcn(parBbeta, mat);
        % mat1 = sparse(idx,ones(1,length(idx))*j,-S_inv*XjtXj*Sinvb+S_inv*Xtxj,N,N);
        % parBalpha = fcn(parBalpha, mat1);

        
        b = (alpha+1)*Xtxj;
        idx = find(non_zero_entries);
        b0 = fun(S, b);
        mat0 = sparse(idx,ones(1,length(idx))*j,b0,N,N);
        B = fcn(B, mat0);

        mat = sparse(idx,ones(1,length(idx))*j,fun(-S*S,b),N,N);
        parBbeta = fcn(parBbeta, mat);

        b1 = XjtXj*b0;
        mat1 = sparse(idx,ones(1,length(idx))*j,fun(-S,b1),N,N);
        parBalpha = fcn(parBalpha, mat1);
        parBalpha = fcn(parBalpha, mat0/(alpha+1));
    end
    % matlabpool close;
    % B = sparse(rows, cols, values, N, N);
    B(1:N+1:end) = 0;
    parBbeta(1:N+1:end) = 0;
    parBalpha(1:N+1:end) = 0;
    P = R;
    Q = B;

end
