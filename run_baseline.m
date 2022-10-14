datasets = {'ml10M','amazon','gowalla','citeulike'};
a = [3.3498,3.8487,3.5491,2.9505,2.7509];
b = [256.5506,256.0611,256.359,256.9426,257.1354];
run_times = 1
for idx = 1:length(datasets)
    dataset = datasets{idx};
    result = [];
    alpha_l = [];
    beta_l = [];
    for i = 1:run_times
        load(['temp/',dataset,'/',int2str(i),'/',dataset,'_ori_split.mat'])
        log_file = ['results/AP_final_results/APEW_five_times/',dataset,'/',int2str(i),'_neighbor_knn.txt'];
        % log_file = 'citulike.txt'

        metric = item_recommend(@iccf, ori_train>0, 'valid', ori_valid, 'test', ori_test, 'topk', 200);
        % [P,Q,metric] = EASE(+(ori_train>0), ori_valid, @tfqmr, 'test', ori_test, 'topk', 200, 'alpha', 10);
        % load(['temp/',dataset,'/',int2str(i),'/NSW.mat'])
        neighber_mat = [];
        alpha = 5;
        beta = 255;
        % [P,Q,metric, a_l, b_l, l_l, r_l, n_l] = EWnLOPT_fast(+(ori_train>0), +(ori_valid>0), @tfqmr, log_file, 'test', ori_test, 'topk', 200, 'alpha', alpha, 'beta', beta,'max_iter', 100, 'lr', 5, 'neighbor', neighber_mat);

        % alpha = metric{2};
        % beta = metric{3};
        % metric = metric{1};
        result=[result;[metric.item_recall(1,10), metric.item_recall(1,20),metric.item_ndcg(1,10), metric.item_ndcg(1,20)]];
        % alpha_l = [alpha_l,alpha];
        % beta_l = [beta_l,beta];
    end
    % result = [result, alpha_l', beta_l'];
    writematrix(result,['results/AP_final_results/WRMF_five_times/WRMF_',dataset,'_1.xls']);
    % writematrix(alpha_l,['results/AP_final_results/APEW_five_times/APEW_',dataset,'_5.xls'],'WriteMode','append');
    % writematrix(beta_l,['results/AP_final_results/APEW_five_times/APEW_',dataset,'_5.xls'],'WriteMode','append');
end
