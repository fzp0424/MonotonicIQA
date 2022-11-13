%4
    Dir = './BID';
    tag = 4;
    
data = xlsread(fullfile(Dir, 'DatabaseGrades.xls'));
name = data(:,1);
mos = data(:, 2);
grades = data(:, 4:end);
mosstd = std(grades, 0, 2, 'omitnan');
    
    imdb.std = mosstd;
    imdb.mos = mos;
    imdb.name = name;
    
    for i = 1:length(name)
        num = name(i);
        if num < 10
            prefix = 'DatabaseImage000';
        elseif num < 100
            prefix = 'DatabaseImage00';
        else
            prefix = 'DatabaseImage0';
        end
        name_t = strcat(prefix, num2str(num), '.JPG');
        all_name{i} = name_t;
    end
    
    imdb.imgpath = all_name;
    
    for split = 1:10
        sel = randperm(586); %6-2-2 586*0.6 = 352 ,586*0.2 = 117
        train_sel = sel(1:352);
        valid_sel = sel(352+1:352+117);
        test_sel = sel(352+117+1:586);
    
        train_path = imdb.imgpath(train_sel);
        valid_path = imdb.imgpath(valid_sel);
        test_path = imdb.imgpath(test_sel);
    
        train_mos = imdb.mos(train_sel);
        valid_mos = imdb.mos(valid_sel);
    
        train_std = imdb.std(train_sel);
        valid_std = imdb.std(valid_sel);


        test_mos = imdb.mos(test_sel);
        test_std = imdb.std(test_sel);
        

        imdb.images.label = [train_mos',valid_mos'];
        imdb.images.std = [train_std',valid_std'];
    
        imdb.classes.description = {'BID'};
        imdb.images.name = [train_path,valid_path] ;
        imdb.imageDir = Dir ;
        

        %for train split
        train_index = 1:length(train_mos);
        fid = fopen(fullfile('./BID/splits2',num2str(split),'bid_train.txt'),'w');
        for i = 1:length(train_index)
            path = fullfile('ImageDatabase',train_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},train_mos(i),train_std(i),tag);
            
        end
        fclose(fid);
        
        %for valid split
        fid = fopen(fullfile('./BID/splits2',num2str(split),'bid_valid.txt'),'w');
        for i = 1:length(valid_path)
            path = fullfile('ImageDatabase',valid_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},valid_mos(i),valid_std(i),tag);
        end
        fclose(fid);

        %for test split
        fid = fopen(fullfile('./BID/splits2',num2str(split),'bid_test.txt'),'w');

        for i = 1:length(test_path)
            path = fullfile('ImageDatabase',test_path(i));
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},test_mos(i),test_std(i),tag);
        end
        fclose(fid);

    end 
    


    disp('BID completed!');
    