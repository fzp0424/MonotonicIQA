%1
    Dir = './databaserelease2';
    tag = 1;
    refpath = fullfile(Dir,'refimgs');
    refpath = strcat(refpath,'/');
    dir_rf = dir([refpath '*.bmp']);
    dmos_t = load(fullfile(Dir,'dmos_realigned.mat'));
    imdb.dmos = dmos_t.dmos_new;
    imdb.orgs = dmos_t.orgs;
    imdb.std = dmos_t.dmos_std;
    
    refname = load(fullfile(Dir,'refnames_all.mat'));
    imdb.refnames_all = refname.refnames_all;
    
    imdb.j2dmos = imdb.dmos(1:227);
    imdb.jpdmos = imdb.dmos(228:460);
    imdb.wndmos = imdb.dmos(461:634);
    imdb.gbdmos = imdb.dmos(635:808);
    imdb.ffdmos = imdb.dmos(809:end);
    
    imdb.j2std = imdb.std(1:227);
    imdb.jpstd = imdb.std(228:460);
    imdb.wnstd = imdb.std(461:634);
    imdb.gbstd = imdb.std(635:808);
    imdb.ffstd = imdb.std(809:end);
    
    imdb.j2orgs = imdb.orgs(1:227);
    imdb.jporgs = imdb.orgs(228:460);
    imdb.wnorgs = imdb.orgs(461:634);
    imdb.gborgs = imdb.orgs(635:808);
    imdb.fforgs = imdb.orgs(809:end);
    
    imdb.orgs = [imdb.j2orgs,imdb.jporgs,imdb.wnorgs,imdb.gborgs,imdb.fforgs];
    
    imdb.refname = cell(1,29);
    for i = 1:29
        file_name = dir_rf(i).name;
        %imdb.refname{i} = string(file_name);
        imdb.refname{i} = file_name;
    end
    
    %%jp2k
    index = 1;
    imdb.dir_j2 = cell(1,227);
    for i = 1:227
        file_name = strcat('img',num2str(i),'.bmp');
        imdb.dir_j2{index} = fullfile('jp2k',file_name);
        index = index + 1;
    end
    
    %%jpeg
    index = 1;
    imdb.dir_jp = cell(1,233);
    for i = 1:233
        file_name = strcat('img',num2str(i),'.bmp');
        imdb.dir_jp{index} = fullfile('jpeg',file_name);
        index = index + 1;
    end
    
    %%white noise
    index = 1;
    imdb.dir_wn = cell(1,174);
    for i = 1:174
           file_name = strcat('img',num2str(i),'.bmp');
           imdb.dir_wn{index} = fullfile('wn',file_name);
           index = index + 1;
    end
    
    %%gblur
    index = 1;
    imdb.dir_gb = cell(1,174);
    for i = 1:174
        file_name = strcat('img',num2str(i),'.bmp');
        imdb.dir_gb{index} = fullfile('gblur',file_name);
        index = index + 1;
    end
    
    %%fast fading
    index = 1;
    imdb.dir_ff = cell(1,174);
    for i = 1:174
        file_name = strcat('img',num2str(i),'.bmp');
        imdb.dir_ff{index} = fullfile('fastfading',file_name);
        index = index + 1;
    end
    
    imdb.imgpath =  cat(2,imdb.dir_j2,imdb.dir_jp,imdb.dir_wn,imdb.dir_gb,imdb.dir_ff);
    imdb.dataset = 'LIVE';
    imdb.filenum = 982;
    
    for split = 1:10
        sel = randperm(29); %6-2-2 6for training 2for validation 2for testing
        train_path = [];
        train_dmos = [];
        train_std = [];
        for i = 1:18
            train_sel = strcmpi(imdb.refname(sel(i)),refname.refnames_all);
            train_sel = train_sel.*(~imdb.orgs);
            train_sel = find(train_sel == 1);
            train_path = [train_path, imdb.imgpath(train_sel)]; 
            train_dmos = [train_dmos,imdb.dmos(train_sel)];
            train_std = [train_std,imdb.std(train_sel)];
        end
    
        valid_path = [];
        valid_dmos = [];
        valid_std = [];
        for i = 19:24
            valid_sel = strcmpi(imdb.refname(sel(i)),refname.refnames_all);
            valid_sel = valid_sel.*(~imdb.orgs);
            valid_sel = find(valid_sel == 1);
            valid_path = [valid_path, imdb.imgpath(valid_sel)]; 
            valid_dmos = [valid_dmos,imdb.dmos(valid_sel)];
            valid_std = [valid_std,imdb.std(valid_sel)];
        end

        test_path = [];
        test_dmos = [];
        test_std = [];
        for i = 25:29
            test_sel = strcmpi(imdb.refname(sel(i)),refname.refnames_all);
            test_sel = test_sel.*(~imdb.orgs);
            test_sel = find(test_sel == 1);
            test_path = [test_path, imdb.imgpath(test_sel)]; 
            test_dmos = [test_dmos,imdb.dmos(test_sel)];
            test_std = [test_std,imdb.std(test_sel)];
        end        
        
        imdb.images.id = 1:870 ;
        imdb.images.label = [train_dmos,valid_dmos,test_dmos];
        imdb.images.std = [train_std,valid_std,test_std];
        imdb.classes.description = {'LIVE'};
        imdb.images.name = [train_path,valid_path,test_path] ;
    
        imdb.images.label = -imdb.images.label + max(imdb.images.label); %the higher the better
        train_length = length(train_dmos);
        valid_length = length(valid_dmos);
        train_dmos = imdb.images.label(1:train_length);
        valid_dmos = imdb.images.label(train_length+1:train_length+valid_length);
        test_dmos = imdb.images.label(train_length+valid_length+1:end);
        
        train_std = imdb.images.std(1:train_length);
        valid_std = imdb.images.std(train_length+1:train_length+valid_length);
        test_std = imdb.images.std(train_length+valid_length+1:end);   

        %for train split
        train_index = 1:length(train_dmos);
        fid = fopen(fullfile('./databaserelease2/splits2',num2str(split),'live_train.txt'),'w');
        for i = 1:length(train_index)
            path = train_path(i);
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},train_dmos(i),train_std(i),tag);
            
        end
        fclose(fid);
        
        %for valid split
        fid = fopen(fullfile('./databaserelease2/splits2',num2str(split),'live_valid.txt'),'w');
        for i = 1:length(valid_path)
            path = valid_path(i); 
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},valid_dmos(i),valid_std(i),tag);
        end
        fclose(fid);
        
        %for test split
        fid = fopen(fullfile('./databaserelease2/splits2',num2str(split),'live_test.txt'),'w');

        for i = 1:length(test_path)
            path = test_path(i);
            path = strrep(path,'\','/');
            fprintf(fid,'%s\t%.3f\t%.3f\t%d\r',path{1},test_dmos(i),test_std(i),tag);
        end
        fclose(fid);

    end




    disp('live completed!');
    