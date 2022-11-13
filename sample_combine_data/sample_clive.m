%5
    Dir = './ChallengeDB_release';
    tag = 5;
    
    imdb.imgpath = cell(1,1162);
    imgpath = fullfile(Dir,'Data','AllImages_release.mat');
    img = load(imgpath);
    img = img.AllImages_release;
    
    mospath = fullfile(Dir,'Data','AllMOS_release.mat');
    mos = load(mospath);
    mos = mos.AllMOS_release;
    imdb.mos = mos(8:end);
    
    stdpath = fullfile(Dir,'Data','AllStdDev_release.mat');
    std = load(stdpath);
    std = std.AllStdDev_release;
    imdb.std = std(8:end);
    
    for i = 8:1169
        file_name = img{i,1};     
        imdb.imgpath{i-7} = fullfile('Images',file_name);
    end
    
    for split = 1:10
        sel = randperm(930+232); %6-2-2 698-232-232
        train_sel = sel(1:698);
        valid_sel = sel(698+1:930);
        test_sel = sel(930+1:1162);
        
        train_path = imdb.imgpath(train_sel);
        valid_path = imdb.imgpath(valid_sel);
        test_path = imdb.imgpath(test_sel);
    
        train_mos = imdb.mos(train_sel);
        valid_mos = imdb.mos(valid_sel);
        
        train_std = imdb.std(train_sel);
        valid_std = imdb.std(valid_sel);



        test_mos = imdb.mos(test_sel);
        test_std = imdb.std(test_sel);
    
        imdb.images.label = [train_mos,valid_mos];
        imdb.images.std = [train_std,valid_std];
    
        imdb.classes.description = {'LIVE_CHAN'};
        imdb.images.name = [train_path,valid_path] ;
        imdb.imageDir = Dir ;
        
        %for train split
        train_index = 1:length(train_mos);

        fid = fopen(fullfile('./ChallengeDB_release/splits2',num2str(split),'clive_train.txt'),'w');
        for i = 1:length(train_index)
            path = train_path(i);
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},train_mos(i),train_std(i),tag);
        end
        fclose(fid);
        
        %for valid split
        fid = fopen(fullfile('./ChallengeDB_release/splits2',num2str(split),'clive_valid.txt'),'w');
        for i = 1:length(valid_path)
            path = valid_path(i);
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},valid_mos(i),valid_std(i),tag);
        end
        fclose(fid);

        %for test split
        fid = fopen(fullfile('./ChallengeDB_release/splits2',num2str(split),'clive_test.txt'),'w');
        for i = 1:length(test_path)
            path = test_path(i);
            path = strrep(path{1,1},'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path,test_mos(i),test_std(i),tag);
        end
        fclose(fid);
    end
        disp('clive completed!');
    