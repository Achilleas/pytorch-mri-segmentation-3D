fp = '../logs/';

%EXP3D_1x1x1x1_0_1_dice_1_log.txt


dilation_arr = [1,1,1,1; 1,2,2,1; 2,2,1,1; 2,2,2,2; 4,4,2,2;4,4,4,4];
priv_arr = [0];
aspp_arr = [0];
loss = 'dice'

count = 1;

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
            filetable
            if priv_str == '0'
                vals = table2array(filetable(:,[3,5,8]))
                iter_num = vals(:,1);
                train_loss = vals(:,2);
                val_loss = vals(:,3);
                
                fig = figure(count)
                hold on
                plot(iter_num, [train_loss, val_loss]);

                legend('Train', 'Val');

                title(title_name);
                xlabel('Iters')
                
                sdf(fig, 'dissertationfigs')
                
                hold off
                count = count + 1;
            else
                %3 iter
                %5 train loss main
                %7 train loss secondary
                %11 val main
                %15 val secondary
                vals = table2array(filetable(:,[3,5,7,11,15]))
                
                iter_num = vals(:,1);
                train_loss_main = vals(:,2);
                train_loss_secondary = vals(:,3);
                
                val_loss_main = vals(:,4);
                val_loss_secondary = vals(:,5)
                
                fig = figure(count)
                hold on
                plot(iter_num, [train_loss_main, val_loss_main]);
                legend('Train', 'Val');
                title(strcat('(Main)', {' '}, title_name));
                xlabel('Iters')
                sdf(fig, 'dissertationfigs')
                
                hold off
                                
                count = count + 1;
                fig = figure(count)
                hold on
                plot(iter_num, [train_loss_secondary, val_loss_secondary]);
                legend('Train', 'Val');
                title(strcat('(Secondary)', {' '} ,title_name));
                xlabel('Iters')
                sdf(fig, 'dissertationfigs')
                
                hold off                                
                count = count + 1;
            end
        end
    end
end