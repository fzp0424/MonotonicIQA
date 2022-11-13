live_root = 'databaserelease2';
csiq_root = 'CSIQ';
tid_root = 'TID2013';
kadid_root = 'kadid10k';

bid_root = 'BID';
clive_root = 'ChallengeDB_release';
koniq_root = 'koniq-10k';

for session = 1:10
    
    filename = fullfile(live_root,'splits2',num2str(session),'live_train.txt');
    fid = fopen(filename);
    live_data=textscan(fid,'%s%f%f%d');
    fclose(fid);
    
    filename = fullfile(csiq_root,'splits2',num2str(session),'csiq_train.txt');
    fid = fopen(filename);
    csiq_data=textscan(fid,'%s%f%f%d');
    fclose(fid);

    filename = fullfile(kadid_root,'splits2',num2str(session),'kadid10k_train.txt');
    fid = fopen(filename);
    kadid_data=textscan(fid,'%s%f%f%d');
    fclose(fid);
    
    filename = fullfile(clive_root,'splits2',num2str(session),'clive_train.txt');
    fid = fopen(filename);
    clive_data=textscan(fid,'%s%f%f%d');
    fclose(fid);
    
    filename = fullfile(bid_root,'splits2',num2str(session),'bid_train.txt');
    fid = fopen(filename);
    bid_data=textscan(fid,'%s%f%f%d');
    fclose(fid);
    
    filename = fullfile(koniq_root,'splits2',num2str(session),'koniq10k_train.txt');
    fid = fopen(filename);
    koniq_data=textscan(fid,'%s%f%f%d');
    fclose(fid);
    
    fid = fopen(fullfile('./splits2',num2str(session),'train.txt'),'w');

    %live
    for i = 1:length(live_data{1,1})
        path = live_data(1);
        mos = live_data(2);
        std = live_data(3);
        tag = live_data(4);
        path = path{1,1};
        mos = mos{1,1};
        std = std{1,1};
        tag = tag{1,1};

        path = fullfile(live_root,path{i,1});
        path = strrep(path, '\', '/');
        fprintf(fid,'%s\t%f\t%f\t%d\r',path, mos(i,1),std(i,1),tag(i,1));
    end
    figure(1)
    mos1 = mos;
    subplot(2,3,1);
    normplot(mos1);
    plot_sigmoid(mos1);
    %[f,xi]=ksdensity(mos1);
    %plot(xi,f);

%     %csiq
    for i = 1:length(csiq_data{1,1})
        path = csiq_data(1);
        mos = csiq_data(2);
        std = csiq_data(3);
        tag = csiq_data(4);
        path = path{1,1};
        mos = mos{1,1};
        std = std{1,1};
        tag = tag{1,1};
        path = fullfile(csiq_root,path{i,1});
        path = strrep(path, '\', '/');
        
        fprintf(fid,'%s\t%f\t%f\t%d\r',path, mos(i,1),std(i,1),tag(i,1));
    end

    mos2 = mos;
    subplot(2,3,2);
    normplot(mos2);
    %[f,xi]=ksdensity(mos2);
    %plot(xi,f);
    
    %kadid
    for i = 1:length(kadid_data{1,1})
        path = kadid_data(1);
        mos = kadid_data(2);
        std = kadid_data(3);
        tag = kadid_data(4);
        path = path{1,1};
        mos = mos{1,1};
        std = std{1,1};
        tag = tag{1,1};
        path = fullfile(kadid_root,path{i,1});
        path = strrep(path, '\', '/');
        
        fprintf(fid,'%s\t%f\t%f\t%d\r',path, mos(i,1),std(i,1),tag(i,1));
    end

    mos3 = mos;
    subplot(2,3,3);
    normplot(mos3);
    %[f,xi]=ksdensity(mos3);
    %plot(xi,f);



   %bid
    for i = 1:length(bid_data{1,1})
        path = bid_data(1);
        mos = bid_data(2);
        std = bid_data(3);
        tag = bid_data(4);
        path = path{1,1};
        mos = mos{1,1};
        std = std{1,1};
        tag = tag{1,1};
        path = fullfile(bid_root,path{i,1});
        path = strrep(path, '\', '/');

        fprintf(fid,'%s\t%f\t%f\t%d\r',path, mos(i,1),std(i,1),tag(i,1));
    end
    mos4 = mos;
    subplot(2,3,4);
    normplot(mos4);
    %[f,xi]=ksdensity(mos4);
    %plot(xi,f);

    %clive
    for i = 1:length(clive_data{1,1})

        path = clive_data(1);
        mos = clive_data(2);
        std = clive_data(3);
        tag = clive_data(4);
        path = path{1,1};
        mos = mos{1,1};
        std = std{1,1};
        tag = tag{1,1};
        path = fullfile(clive_root,path{i,1});
        path = strrep(path, '\', '/');
        
        fprintf(fid,'%s\t%f\t%f\t%d\r',path, mos(i,1),std(i,1),tag(i,1));
    end

    mos5 = mos;
    subplot(2,3,5);
    normplot(mos5);
    %[f,xi]=ksdensity(mos5);
    %plot(xi,f);
    

    %koniq-10k
    for i = 1:length(koniq_data{1,1})
        path = koniq_data(1);
        mos = koniq_data(2);
        std = koniq_data(3);
        tag = koniq_data(4);
        path = path{1,1};
        mos = mos{1,1};
        std = std{1,1};
        tag = tag{1,1};
        path = fullfile(koniq_root,path{i,1});
        path = strrep(path, '\', '/');
        
        fprintf(fid,'%s\t%f\t%f\t%d\r',path, mos(i,1),std(i,1),tag(i,1));
    end
    mos6 = mos; 
    subplot(2,3,6);
    normplot(mos6);
    %[f,xi]=ksdensity(mos6);
    %plot(xi,f);


end

disp('combine completed!');


