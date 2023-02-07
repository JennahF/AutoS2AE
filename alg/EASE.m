function [P, Q, metric] = EASE(R, R_v, fun, varargin)
[M, N] = size(R);


[alpha, test, beta] = process_options(varargin, 'alpha', 10, 'test', [],'beta', 250);


fprintf('alpha=%f, beta=%f)\n', alpha, beta);
    
P = [];
Q = [];
fprintf('Start calculating')
tic;[P, Q] = ease(R, alpha, beta, fun);toc;

fprintf('Start testing');

if (~isempty(test))
    tic;metric = evaluate_item(R+R_v, test, P, Q.', 200, 200);toc;
    if isexplict(test)
        fprintf('\nrecall@10,20,50=%.3f,%.3f,%.3f', metric.item_recall_like(1,10), metric.item_recall_like(1,20), metric.item_recall_like(1,50));
        fprintf('\nndcg@10,20,50=%.3f,%.3f,%.3f', metric.item_ndcg_like(1,10), metric.item_ndcg_like(1,20), metric.item_ndcg_like(1,50));
    else
        fprintf('\nrecall@10,20,50=%.3f,%.3f,%.3f', metric.item_recall(1,10), metric.item_recall(1,20), metric.item_recall(1,50));
        fprintf('\nndcg@10,20,50=%.3f,%.3f,%.3f', metric.item_ndcg(1,10), metric.item_ndcg(1,20), metric.item_ndcg(1,50));
    end
end
fprintf('\n');
end

function [x] = chebyshev_semi_iterate(G,c)

    beta = 0.5;
    alpha = -2;
    epsilon = 0.001;
    [m,n] = size(G);
    x0 = ones(n,1);
    rho1 = 2;
    x1 = G*x0 + c;
    xi = (2-beta-alpha)/(beta-alpha);
    nu = 2/(2-beta-alpha);

    while(norm((x0-x1),2)>epsilon)
        rho2=1/(1-(rho1/4/xi^2));
        x2 = rho2*(nu.*(G*x1+c)+(1-nu).*x1)+(1-rho2)*x0;
        rho1 = rho2;
        x0 = x1;
        x1 = x2;
    end
    x = x1;

end


function [P, Q] = ease(R, alpha, beta, fun)

    [M,N] = size(R);

    G = R'*R;
    G(logical(eye(size(G))))=diag(G)+beta;
    P = inv(G);
    B = P/diag(-diag(P));
    B(1:N+1:end) = 0;
    P = R;
    Q = B;

end
