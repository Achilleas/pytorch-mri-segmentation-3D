%aspp vs no aspp for 1x1x1x1
fp = '../logs/';

%EXP3D_1x1x1x1_0_1_dice_1_log.txt

dilation_arr = [1,1,1,1;];
priv_arr = [0];
aspp_arr = [0;1];
loss = 'dice';

for d_i = 1:size(dilation_arr, 1)
    for p_i = 1:size(priv_arr, 1)
        for a_i = 1:size(aspp_arr, 1)
            dilation_str = strrep(num2str(dilation_arr(d_i, :)), '  ', 'x');
            priv_str = num2str(priv_arr(p_i));
            aspp_str = num2str(aspp_arr(a_i));
            fname = strcat(fp, 'EXP3D_', dilation_str, '_', priv_str, '_', aspp_str, '_', loss, '_1_log.txt');
            title_name = strcat('Dilations: ', dilation_str, ' withPriv = ', priv_str, ' withASPP = ', aspp_str);
            fname = char(fname);

            filetable = readtable(fname, 'Delimiter', ' ');
            
            if priv_str == '0'
                vals = table2array(filetable(:,[3,5,8]));
                iter_num = vals(:,1);
                train_loss = vals(:,2);
                val_loss = vals(:,3);
                
                fig = figure(1);
                hold on
                plot(iter_num, val_loss);
            end
        end
    end
end

legend('1x1x1x1 no aspp', '1x1x1x1 with aspp');
title(title_name);
xlabel('Iters')
sdf(fig, 'dissertationfigs')
hold off