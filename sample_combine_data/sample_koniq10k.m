%6
    Dir = './koniq-10k';
    fileID = fopen(fullfile(Dir,'koniq10k_scores_and_distributions.csv'));
    % load(fullfile(Dir,'koniq10k_scores_and_distributions.mat'));
    %data = koniq10kscoresanddistributions;
    %data = csvread(fullfile(Dir,'koniq10k_scores_and_distributions.csv'));
    data = textscan(fileID,'%s %f %f %f %f %f %f %f %f %s\n','HeaderLines',1,'Delimiter',',','EndOfLine', '\n');
    tag = 6;
    
    %num = 10073
    imagename = data(:,1);
    mos_original = data(:,8);
    std_original = data(:,9);

    imagename = imagename{1,1};
    mos_original = mos_original{1,1};
    std_original = std_original{1,1};

    mos = zeros(1,10073,'single');
    std = zeros(1,10073,'single');
    
    for i = 1:10073
        mos(1,i) = single(mos_original(i));
        std(1,i) = single(std_original(i));
    end
    
    for split = 1:10
        sel = randperm(6043+2015+2015);%6-2-2 6043 2015 2015
        train_sel = sel(1:6043);
        valid_sel = sel(6043+1:6043+2015);
        test_sel = sel(6043+2015+1:10073);
        train_path = imagename(train_sel);
        valid_path = imagename(valid_sel);
        test_path = imagename(test_sel);
        train_mos = mos(train_sel);
        valid_mos = mos(valid_sel);
        test_mos = mos(test_sel);
        train_std = std(train_sel);
        valid_std = std(valid_sel);
        test_std = std(test_sel);
         
        %for train split
        train_index = 1:length(train_mos);        
        fid = fopen(fullfile('./koniq-10k/splits2/',num2str(split),'koniq10k_train.txt'),'w');
        for i = 1:length(train_index)
            path = fullfile('1024x768',train_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},train_mos(i),train_std(i),tag);
        end
        fclose(fid);
        
        %for valid split
        fid = fopen(fullfile('./koniq-10k/splits2',num2str(split),'koniq10k_valid.txt'),'w');
        for i = 1:length(valid_path)
            path = fullfile('1024x768',valid_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},valid_mos(i),valid_std(i),tag);
        end
        fclose(fid);
        
        %for test split
        fid = fopen(fullfile('./koniq-10k/splits2',num2str(split),'koniq10k_test.txt'),'w');
        for i = 1:length(test_path)
            path = fullfile('1024x768',test_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},test_mos(i),test_std(i),tag);
        end
        fclose(fid);   
    end
    fclose(fileID);
    disp('koniq10k completed!');
    