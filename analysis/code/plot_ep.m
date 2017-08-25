%Plot validation scores over all dilation files (no priv, aspp = 1)
fp = '../logs/';

%1x1x1x1_0_1 0 iou: 0.6958 dice: 0.5479 recall: 0.5284 precision: 0.6050
%ep_2x2x2x2.txt

dilation_arr = [1,1,1,1; 2,2,2,2];
priv_arr = [0];
aspp_arr = [1];
loss = 'dice';

fig = figure(1);
hold on
for d_i = 1:size(dilation_arr, 1)
    for p_i = 1:size(priv_arr, 1)
        for a_i = 1:size(aspp_arr, 1)
            dilation_str = strrep(num2str(dilation_arr(d_i, :)), '  ', 'x');
            priv_str = num2str(priv_arr(p_i));
            aspp_str = num2str(aspp_arr(a_i));
            fname = strcat(fp, 'ep_', dilation_str, '_', priv_str, '_', aspp_str, '_8.txt');
            title_name = strcat('Dilations: ', ' withPriv = ', priv_str, ' withASPP = ', aspp_str);
            fname = char(fname);

            filetable = readtable(fname, 'Delimiter', ' ');
            vals = table2array(filetable(:,[2,4,6,8]));
            iter_num = vals(:,1);
            iou = vals(:,2);
            plot(iter_num, iou);
        end
    end
end

legend('1x1x1x1', '2x2x2x2');
title(title_name);
xlabel('Extra patch size (ep)')
sdf(fig, 'dissertationfigs')
hold off