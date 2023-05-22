datasets = {'citeulike', 'ml10M', 'gowalla', 'amazon'};
run_times = 1
for idx = 1:length(datasets)
    dataset = datasets{idx};
    result = [];
    alpha_l = [];
    beta_l = [];
    for i = 1:run_times
        load(['temp/',dataset,'/',int2str(i),'/',dataset,'_ori_split.mat'])
        log_file = ['results/',dataset,'/',int2str(i),'.txt'];
        % log_file = 'citulike.txt'

	    %----WRMF----:
        % metric = item_recommend(@iccf, ori_train>0, 'valid', ori_valid, 'test', ori_test, 'topk', 200);
        
        %----EASE----;
        % [P,Q,metric] = EASE(+(ori_train>0), ori_valid, @tfqmr, 'test', ori_test, 'topk', 200, 'alpha', 10);
        
        %----AutoS2AE----;
        % load(['temp/',dataset,'/',int2str(i),'/NSW.mat'])
        neighber_mat = [];
        alpha = 5;
        beta = 255;
        [P,Q,metric, a_l, b_l, l_l, v_l, r_l, n_l] = EWnLOPT_fast(+(ori_train>0), +(ori_valid>0), @tfqmr, log_file, 'test', ori_test, 'topk', 200, 'alpha', alpha, 'beta', beta,'max_iter', 3, 'lr', 5, 'neighbor', neighber_mat);

        result=[result;[metric.item_recall(1,10), metric.item_recall(1,20),metric.item_ndcg(1,10), metric.item_ndcg(1,20)]];
        
    end
    writematrix(result,['results/AutoS2AE_',dataset,'_1.xls']);
end
