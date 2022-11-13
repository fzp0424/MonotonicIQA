%3
    Dir = './kadid10k';
    fileID = fopen(fullfile(Dir, 'dmos.csv'));
    data = textscan(fileID,'%s %s %f %f\n', 'HeaderLines',1,'Delimiter',',');
    tag = 3;
    
    imagename = data(:,1);
    refnames_all = data(:,2);
    mos_original = data(:,3);
    std_original = data(:,4);
    
    imagename = imagename{1,1};
    refnames_all = refnames_all{1,1};
    mos_original = mos_original{1,1};
    std_original = std_original{1,1};
    
    mos_original = mos_original';
    std_original = sqrt(std_original');
    
    mos = zeros(1,10125,'single');
    std = zeros(1,10125,'single');
    for i = 1:10125
        mos(1,i) = single(mos_original(i));
        std(1,i) = single(std_original(i));
    end
    
    
    refname = refnames_all(1:125:end); %81 refimgs in total, each refimg has 125 distotred imgs
    
    for split = 1:10
        sel = randperm(81); %6-2-2 ---49 for training, 16 for validation, 16 for testing
        train_path = [];
        train_mos = [];
        train_std = [];
        for i = 1:49
            train_sel = strcmpi(refname(sel(i)),refnames_all );
            train_sel = find(train_sel == 1);
            train_path = [train_path, imagename(train_sel)']; 
            train_mos = [train_mos,mos_original(train_sel)];
            train_std = [train_std,std_original(train_sel)];
        end
    
        valid_path = [];
        valid_mos = [];
        valid_std = [];
        for i = 50:65
            valid_sel = strcmpi(refname(sel(i)),refnames_all );
            valid_sel = find(valid_sel == 1);
            valid_path = [valid_path, imagename(valid_sel)']; 
            valid_mos = [valid_mos,mos_original(valid_sel)];
            valid_std = [valid_std,std_original(valid_sel)];
        end
        
        test_path = [];
        test_mos = [];
        test_std = [];
        for i = 66:81
            test_sel = strcmpi(refname(sel(i)),refnames_all);
            test_sel = find(test_sel == 1);
            test_path = [test_path, imagename(test_sel)']; 
            test_mos = [test_mos,mos_original(test_sel)];
            test_std = [test_std,std_original(test_sel)];
        end

         
        %for train split
        train_index = 1:length(train_mos);
        fid = fopen(fullfile('./kadid10k/splits2/',num2str(split),'kadid10k_train.txt'),'w');
        for i = 1:length(train_index)
            path = fullfile('images',train_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},train_mos(i),train_std(i),tag); 
        end
        fclose(fid);
        
        %for valid split
        fid = fopen(fullfile('./kadid10k/splits2',num2str(split),'kadid10k_valid.txt'),'w');
        for i = 1:length(valid_path)
            path = fullfile('images',valid_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},valid_mos(i),valid_std(i),tag);
        end
        fclose(fid);

        %for test split
        fid = fopen(fullfile('./kadid10k/splits2',num2str(split),'kadid10k_test.txt'),'w');
        for i = 1:length(test_path)
            path = fullfile('images',test_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},test_mos(i),test_std(i),tag);
        end
        fclose(fid);
    end
    fclose(fileID);

    
    disp('kadid10k completed!');
    