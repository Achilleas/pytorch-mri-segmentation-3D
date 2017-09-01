%Plot priv info vs not priv info for validation
fp = '../logs/';

%EXP3D_1x1x1x1_0_1_dice_1_log.txt
dilation_arr = [1,1,1,1;];
priv_arr = [1];
aspp_arr = [1];
names = [""; "trainalt"];
loss = 'dice';
legends = ["1x1x1x1 main","1x1x1x1 secondary", "1x1x1x1 alt main", "1x1x1x1 alt secondary"];
count = 1;
legcount = 1;

for d_i = 1:size(dilation_arr, 1)
    fig = figure(count)
    hold on
    for p_i = 1:size(priv_arr, 1)
        for a_i = 1:size(aspp_arr, 1)
            for n_i = 1:size(names, 1)
                dilation_str = strrep(num2str(dilation_arr(d_i, :)), '  ', 'x');
                priv_str = num2str(priv_arr(p_i));
                aspp_str = num2str(aspp_arr(a_i));
                fname = strcat(fp, 'EXP3D', names(n_i), '_', dilation_str, '_', priv_str, '_', aspp_str, '_', loss, '_1_log.txt');
                title_name = strcat('Dilations: ', dilation_str, ' withASPP = ', aspp_str);
                if priv_str == '0' && n_i == 2
                    continue
                end
                fname = char(fname);
                filetable = readtable(fname, 'Delimiter', ' ');
                if priv_str == '0'
                    vals = table2array(filetable(:,[3,5,8]));
                    iter_num = vals(:,1);
                    train_loss = vals(:,2);
                    val_loss = vals(:,3);
                    
                    plot(iter_num, train_loss);
                else
                    vals = table2array(filetable(:,[3,5,7,11,15]));

                    iter_num = vals(:,1);
                    train_loss_main = vals(:,2);
                    train_loss_secondary = vals(:,3);

                    val_loss_main = vals(:,4);
                    val_loss_secondary = vals(:,5);
                    
                    plot(iter_num, train_loss_main);   
                    plot(iter_num, train_loss_secondary);                
                end
            end
        end
    end
    count = count + 1;
    legend(legends);
    title(strcat('1x1x1x1 main and secondary losses'));
    xlabel('Iters');
    sdf(fig, 'dissertationfigs')
    hold off
end
