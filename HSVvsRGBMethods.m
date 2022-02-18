
%By using this code the binarized images are created.

%Part of this code is based in code created by Carlos López-Molina (see The kermit image toolkit 
%(kitt), ghent university, www.kermitimagetoolkit.net.) Some functions that are being called during the code 
%can be found at kermit's webpage as well (i.e. Canny(), Sobel(), etc.). Anyway, other equivalent functions could be used 
%and almost the same results are expected. The only requirement is that the function of this Edge detection algorithms
%allows keeping the soft value before scaling is made. 

%Here you need to change the folder to suit to your needs".
ROOTFOLDER='C:\Users\Edge Detection\...';

Experiment.orgDir = strcat(ROOTFOLDER,'Resource\BSDS500\images\train\');  

% Hand-made ground-truth (solutions)
Experiment.gtOrgDir=strcat(ROOTFOLDER,'Resource\BSDS500\groundTruth\train_Total Human\');
    
% Folder to store the final results after binarization
Experiment.bnDir=strcat(ROOTFOLDER,'train\BN\');
 
%Folder to store the quantitative values yielded from the comparison of the
%automatic solutions to the hand-made ones
Experiment.cpDir=strcat(ROOTFOLDER,'train\Comparison');
  
%Range of the images (in the folder) to be selected for the comparison
Experiment.imagesFrom1=1;
Experiment.imagesTo1=24;
Experiment.imagesFrom2=25;
Experiment.imagesTo2=50;
%Experiment.imagesFrom3=41;
%Experiment.imagesTo3=50;
Experiment.trainsize=50;
Experiment.testsize=20;
%Experiment.numsamples=3;
     
else
    error('Wrong Experiment.databaseName at infoMaker.m');
end
    
%Extension (coding) of the original images
Experiment.orgImagesExt='jpg'; 
Experiment.gtImagesExt='png'; 
%Extension (coding) of the automatically generated images
Experiment.imagesExt='png';
%Extension of the matlab/octave data files
Experiment.dataExt='mat';
%Prefix in the name of the grayscale images
Experiment.gsPrefix='gs-';
%Prefix in the name of the binary edge images
Experiment.bnPrefix='bn-';
%Prefix in the name of the files containing the comparison values
Experiment.cpPrefix='cp-';

% The following fields force the experiments to be repeated, 
% even if the results have been stored in a previous launch 
Experiment.forceBnMaker = 1;
Experiment.forceComparer = 0;

% Range of the sigmas in the zero-th (smoothing) and first (differentiation) order. Gaussian kernels. 
Experiment.SigmasOne = [1.0]; %The Gaussian smoothing value. 
Experiment.cannySigmasTwo = [2.0]; %[0.5:0.5:2.5]; The Canny's sigma. 

% Strategies for binarization
% Here is where you add up names of functions able to do the clustering

Experiment.methods = {'sobelgray','sobelAndrews','SobDifT&FuMax','FuzzNYager','FuzzDirNYager','PostNYagerMax','sobelNYager','SobMaxAgreg','SobMaxDirAgreg','CannyNorm','CAndrews','CMaxNAgreg','CMeanAgreg','CannYager','PostCannyYagerMax','FuzzCanYager','FuzzDirCYager'};

% Here are specified the parameters of the different methods
Experiment.canny.gray = 0.1:0.05:0.90; 
Experiment.sobel.gray = 0.1:0.05:0.90; 
Experiment.sobel.red = 0.15:0.05:0.80; 
Experiment.sobel.green = 0.15:0.05:0.80; 
Experiment.sobel.blue = 0.15:0.05:0.80;

% This part is devoted to the matching process
Experiment.EMs={'ejBCM-F_5'};
Experiment.matchingDistance=5;
Experiment.PrecisionRankingPos=1;%The position of the EM after which PR is selected
Experiment.RecallRankingPos=1;%The position of the EM after which RC is selected

%Range of the images (in the folder) to be selected for the comparison
Experiment.imagesFrom1=1;
Experiment.imagesTo1=24;
Experiment.imagesFrom2=25;
Experiment.imagesTo2=50;
%Experiment.imagesFrom3=41;
%Experiment.imagesTo3=50;
Experiment.trainsize=50;

imagesList=getFileList(Experiment.orgDir,'*',Experiment.orgImagesExt);
imagesList1=imagesList(Experiment.imagesFrom1:Experiment.imagesTo1);
imagesList2=imagesList(Experiment.imagesFrom2:Experiment.imagesTo2);
imagesList =[imagesList1 imagesList2];
trainsize=Experiment.trainsize;
imagesList=imagesList(1:trainsize);

% Here is where the binarization code starts

for idxImagesList=1:length(imagesList)

    %Reading the Image
    fullImageName=char(imagesList(idxImagesList));
    rawImageName=regexprep(fullImageName,strcat('.',Experiment.orgImagesExt),'');
    imagePath=strcat(Experiment.orgDir,fullImageName);
    
    %Normalizing the image
    imginteger=double(imread(imagePath));
    if (max(imginteger(:))>1)
        img=imginteger./255;
    end
    
    %Create the output folder for this image
	imageBnFolder=strcat(Experiment.bnDir,rawImageName,'/');
    if (~exist(imageBnFolder,'dir'))
		mkdir(imageBnFolder);
    end
    
    %For each std. dev a different smoothing is applied.
    for idSigmaOne=1:length(Experiment.SigmasOne)
        sigmaOne=Experiment.SigmasOne(idSigmaOne);
        sigmaOneName=regexprep(sprintf('%.2f',sigmaOne),'\.','-');
        
        if sigmaOne>=0.6 
           smooImg=gaussianSmooth(img,sigmaOne); 
        else
           smooImg=img; 
        end	
        
        %% For each Binarization method
        for idMethod=1:length(Experiment.methods)

            methodName=char(Experiment.methods(idMethod))
            tic;    
            bnFileName=strcat(Experiment.bnPrefix,createName(rawImageName,sigmaOneName,methodName));
            bnFilePath=strcat(Experiment.bnDir,rawImageName,'\');

            %% First, conventional
            if ((~exist(bnFilePath,'file')) || (Experiment.forceBnMaker==1))
            
                if  (contains(methodName,'CannyNorm'))
                    if (size(smooImg,3)>1)
                        smooImgGrey=mean(smooImg,3);
                    else
                        smooImgGrey = smooImg;    
                    end
                    
                    for idSigmaTwo=1:length(Experiment.cannySigmasTwo)

                        sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                        sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-');    
                        [fx,fy]=canny(smooImgGrey,sigmaTwo);
                        ft=sqrt(fx.^2+fy.^2);
                        fx=fx./max(ft(:));
                        fy=fy./max(ft(:));
                        ft=ft./max(ft(:));
                        [maxMap]=directionalNMS(fx,fy);
                        [cleanfx,cleanfy]=directionalNMS(fx,fy);
                        ft2=ft.*maxMap;

                        for idthr=1:length(Experiment.canny.gray)

                            thrHigh=Experiment.canny.gray(idthr);
                            thrString=sprintf('%02d',round(thrHigh*100));
                            thrLow=thrHigh.*0.4;
                            bnImage=floodHysteresis(ft2,thrHigh,thrLow);
                            bnFileName=strcat(Experiment.bnPrefix,...                                  
                            createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                            bnFileNameComp=strcat(bnFileName,'-',num2str(thrString),'.',Experiment.imagesExt);
                            imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));                                 
                        end
                        fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);                   
                    end
                                                      
                elseif  (contains(methodName,'CAndrews'))
                            
                            for idSigmaTwo=1:length(Experiment.cannySigmasTwo)

                                sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                                sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-');

                                for channel=1:3

                                    [fx,fy]=canny(smooImg(:,:,channel),sigmaTwo);
                                    ft=sqrt(fx.^2+fy.^2);
                                    fx=fx./max(ft(:));
                                    fy=fy./max(ft(:));
                                    ft=ft./max(ft(:));
                                    maxMap=directionalNMS(fx,fy);
                                    [cleanfx,cleanfy]=directionalNMS(fx,fy);
                                    ft2(:,:,channel)=ft.*maxMap;    
                                end
                                for idthr=1:length(Experiment.canny.gray)

                                    thrHigh=Experiment.canny.gray(idthr);
                                    thrString=sprintf('%02d',round(thrHigh*100));
                                    thrLow=thrHigh.*0.4;
                                    bnImage(:,:,1) = floodHysteresis(ft2(:,:,1),thrHigh,thrLow);
                                    bnImage(:,:,2) = floodHysteresis(ft2(:,:,2),thrHigh,thrLow);
                                    bnImage(:,:,3) = floodHysteresis(ft2(:,:,3),thrHigh,thrLow);
                                    bnImageMax(:,:) = max(bnImage,[],3);

                                    bnFileName=strcat(Experiment.bnPrefix,...                                  
                                    createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                                    bnFileNameComp=strcat(bnFileName,'-',num2str(thrString),'.',Experiment.imagesExt);
                                    imwrite(bnImageMax,strcat(bnFilePath,bnFileNameComp));                                 
                                end
                                fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);
                            end
                            clear ft2 bnImage bnImageMax, maxMap;
                   

                elseif  (contains(methodName,'CMeanAgreg'))

                            for idSigmaTwo=1:length(Experiment.cannySigmasTwo)
                                sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                                sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-');
                                
                                for channel=1:3

                                    [fx,fy]=canny(smooImg(:,:,channel),sigmaTwo);
                                    ft=sqrt(fx.^2+fy.^2);
                                    fx=fx./max(ft(:));
                                    fy=fy./max(ft(:));
                                    ft=ft./max(ft(:));
                                    maxMap=directionalNMS(fx,fy);
                                    ft2(:,:,channel)=ft.*maxMap;
                                    clear fx fy ft;
                                end
                                
                                for idthr=1:length(Experiment.canny.gray)

                                    thrHigh=Experiment.canny.gray(idthr);
                                    thrString=sprintf('%02d',round(thrHigh*100));
                                    thrLow=thrHigh.*0.4;
                                    ft2mean=mean(ft2,3);
                                    bnImage=floodHysteresis(ft2mean,thrHigh,thrLow);
                                    bnFileName=strcat(Experiment.bnPrefix,...                                  
                                        createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                                    bnFileNameComp=strcat(bnFileName,'-',num2str(thrString),'.',Experiment.imagesExt);
                                    imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));                                 
                                end
                                
                                fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);
                            end                                        

                elseif  (contains(methodName,'CMaxNAgreg'))
                            
                        for idSigmaTwo=1:length(Experiment.cannySigmasTwo)
                                
                            sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                            sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-');
                                
                            for channel=1:3

                                [fx,fy]=canny(smooImg(:,:,channel),sigmaTwo);
                                ft=sqrt(fx.^2+fy.^2);
                                fx=fx./max(ft(:));
                                fy=fy./max(ft(:));
                                ft=ft./max(ft(:));
                                maxMap=directionalNMS(fx,fy);
                                ft2(:,:,channel)=ft.*maxMap;
                                clear fx fy ft;
                            end
                                
                            for idthr=1:length(Experiment.canny.gray)

                                thrHigh=Experiment.canny.gray(idthr);
                                thrString=sprintf('%02d',round(thrHigh*100));
                                thrLow=thrHigh.*0.4;
                                ft2max=max(ft2,[],3);
                                bnImage=floodHysteresis(ft2max,thrHigh,thrLow);
                                bnFileName=strcat(Experiment.bnPrefix,...                                  
                                        createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                                bnFileNameComp=strcat(bnFileName,'-',num2str(thrString),'.',Experiment.imagesExt);
                                imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));                                 
                            end
                            fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);
                        end

                elseif (strcmp(methodName,'CannYager'))                
                    
                    if (size(smooImg,3)==3) %If the image is a RGB image do...
                        smooImgHSV = rgb2hsv(smooImg);
                    end
                        
                    for idSigmaTwo=1:length(Experiment.cannySigmasTwo)

                        sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                        sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-');    

                        for w1 = 0:0.1:1

                            restow1 = 1-w1; 

                            for w2 = 0:0.1:restow1  

                                ImgHSVYager=HSVYager(smooImgHSV,w1,w2,1-(w1+w2)); %The HSV image (3 channels) is aggregated into a single channel                                
                                [ft,fx,fy]=canny(ImgHSVYager,sigmaTwo);
                                fx=fx./max(ft(:));
                                fy=fy./max(ft(:));
                                ft=ft./max(ft(:));
                                maxMap=directionalNMS(fx,fy); 
                                ft2=ft.*maxMap;
                            
                                for idthr=1:length(Experiment.canny.gray)

                                    thrHigh=Experiment.canny.gray(idthr);
                                    thrString=sprintf('%02d',round(thrHigh*100));
                                    w1String=sprintf('%02d',round(w1*100));
                                    w2String=sprintf('%02d',round(w2*100));
                                    w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                    thrLow=thrHigh.*0.4;
                                    bnImage=floodHysteresis(ft2,thrHigh,thrLow);
                                    bnFileName=strcat(Experiment.bnPrefix,...                                  
                                    createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                                    bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                    imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                                end

                            clear ft2;
                            clear bnImage;
                            clear maxMap;

                            end  
                        end
                    fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);                     
                    end
                                                
                elseif (strcmp(methodName,'sobelgray'))                    
                      
                    if (size(smooImg,3)>1)
                        smooImg=mean(smooImg,3); %this aggregates the three RGB channels into one, that now is grayscale (0 to 255)
                    end
                        
                    [ft,fx,fy]=sobel(smooImg);
                    fx=fx./max(ft(:));
                    fy=fy./max(ft(:));
                    ft=ft./max(ft(:));
                    maxMap=directionalNMS(fx,fy); 
                    ft2=ft.*maxMap;
                        
                    for idthr=1:length(Experiment.sobel.gray)

                        thr=Experiment.sobel.gray(idthr);
                        thrString=sprintf('%02d',round(thr*100));
                        bnImage=im2bw(ft2,thr);
                        bnFileNameComp=strcat(bnFileName,num2str(thrString),'.',Experiment.imagesExt);
                        imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                    end

                    fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                    clear ft2;
                    clear bnImage;
                    clear maxMap;

                elseif  (contains(methodName,'sobelAndrews'))

                        for channel=1:3

                            [fx,fy]=sobel(smooImg(:,:,channel));
                            ft=sqrt(fx.^2+fy.^2);
                            fx=fx./max(ft(:));
                            fy=fy./max(ft(:));
                            ft=ft./max(ft(:));
                            maxMap=directionalNMS(fx,fy);
                            [cleanfx,cleanfy]=directionalNMS(fx,fy);
                            ft2(:,:,channel)=ft.*maxMap;    
                        end
                        
                        for idthr=1:length(Experiment.sobel.gray)

                            thr=Experiment.sobel.gray(idthr);
                            thrString=sprintf('%02d',round(thr*100));
                            bnImage(:,:,1) = im2bw(ft2(:,:,1),thr);
                            bnImage(:,:,2) = im2bw(ft2(:,:,2),thr);
                            bnImage(:,:,3) = im2bw(ft2(:,:,3),thr);
                            bnImageMax(:,:) = max(bnImage,[],3);
                            bnFileNameComp=strcat(bnFileName,num2str(thrString),'.',Experiment.imagesExt);
                            imwrite(bnImageMax,strcat(bnFilePath,bnFileNameComp));                            
                        end
                        fprintf('\t Done for sigmaOne=%.2f,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);
                        
                        clear ft2 bnImage bnImageMax, maxMap;
             

                elseif (strcmp(methodName,'sobelNYager'))                
                    
                    if (size(smooImg,3)==3) %If the image is a RGB image do...
                        smooImgHSV = rgb2hsv(smooImg);
                    end

                    for w1 = 0:0.1:1

                        restow1 = 1-w1; 

                        for w2 = 0:0.1:restow1 

                            ImgHSVYager=HSVYager(smooImgHSV,w1,w2,1-(w1+w2)); %The HSV image (3 channels) is aggregated in one single channel                            
                            [ft,fx,fy]=sobel(ImgHSVYager);
                            fx=fx./max(ft(:));
                            fy=fy./max(ft(:));
                            ft=ft./max(ft(:));
                            maxMap=directionalNMS(fx,fy); 
                            ft2=ft.*maxMap;
                                
                            for idthr=1:length(Experiment.sobel.gray)

                                thr=Experiment.sobel.gray(idthr);
                                thrString=sprintf('%02d',round(thr*100));
                                w1String=sprintf('%02d',round(w1*100));
                                w2String=sprintf('%02d',round(w2*100));
                                w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                bnImage=imbinarize(ft2,thr);
                                bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                            end
                            
                            clear ft2;
                            clear bnImage;
                            clear maxMap;
                                    
                        end
                    end

                    fprintf('\t Done for sigmaOne=%.2f, binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                elseif (strcmp(methodName,'PostNYagerMax'))

                        smooImgV=zeros(size(smooImg(:,:,1)));                    
                    
                        if (size(smooImg,3)==3) %If the image is a RGB image do...
                           smooImgHSV=rgb2hsv(smooImg);
                        end

                        for w1 = 0:0.1:1

                            restow1 = 1-w1; 

                            for w2 = 0:0.1:restow1 
                                    
                                ImgHSVYager(:,:,1)=HSVYager(smooImgHSV,w1,0,0);
                                ImgHSVYager(:,:,2)=HSVYager(smooImgHSV,0,w2,0);
                                ImgHSVYager(:,:,3)=HSVYager(smooImgHSV,0,0,1-(w1+w2));
                                    
                                for channel=1:3

                                    [ft,fx,fy]=sobel(ImgHSVYager(:,:,channel));
                                    fx=fx./max(ft(:));
                                    fy=fy./max(ft(:));
                                    ft=ft./max(ft(:));
                                    maxMap=directionalNMS(fx,fy);
                                    ft2(:,:,channel)=ft.*maxMap;
                                    clear maxMap ft fx fy;
                                end
                            
                                for idthr=1:length(Experiment.sobel.gray)

                                    thr=Experiment.sobel.gray(idthr);
                                    thrString=sprintf('%02d',round(thr*100));
                                    w1String=sprintf('%02d',round(w1*100));
                                    w2String=sprintf('%02d',round(w2*100));
                                    w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                    bnImage(:,:,1)=im2bw(ft2(:,:,1),thr);
                                    bnImage(:,:,2)=im2bw(ft2(:,:,2),thr);
                                    bnImage(:,:,3)=im2bw(ft2(:,:,3),thr);
                                    bnImageMaxAgreg(:,:)=max(bnImage,[],3);
                                    bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                    imwrite(bnImageMaxAgreg,strcat(bnFilePath,bnFileNameComp));
                                end
                                    
                                clear ft2;
                                clear bnImage;
                                clear bnImageMaxAgreg;
                                clear maxMap;
                            end
                        end

                        fprintf('\t Done for sigmaOne=%.2f, binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                elseif (strcmp(methodName,'PostCannyYagerMax'))

                        smooImgV=zeros(size(smooImg(:,:,1)));                    
                    
                        if (size(smooImg,3)==3) %If the image is a RGB image do...
                           smooImgHSV=rgb2hsv(smooImg);
                        end

                        for idSigmaTwo=1:length(Experiment.cannySigmasTwo)

                            sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                            sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-'); 

                            for w1 = 0:0.1:1
                            
                                restow1 = 1-w1; 
                            
                                for w2 = 0:0.1:restow1 
                                    
                                    ImgHSVYager(:,:,1)=HSVYager(smooImgHSV,w1,0,0);
                                    ImgHSVYager(:,:,2)=HSVYager(smooImgHSV,0,w2,0);
                                    ImgHSVYager(:,:,3)=HSVYager(smooImgHSV,0,0,1-(w1+w2));

                                    for channel=1:3

                                        [ft,fx,fy]=canny(ImgHSVYager(:,:,channel),sigmaTwo);
                                        fx=fx./max(ft(:));
                                        fy=fy./max(ft(:));
                                        ft=ft./max(ft(:));
                                        maxMap=directionalNMS(fx,fy);
                                        ft2(:,:,channel)=ft.*maxMap;
                                        clear maxMap ft fx fy;
                                    end
                            
                                        for idthr=1:length(Experiment.canny.gray)

                                            thrHigh=Experiment.canny.gray(idthr);
                                            thrString=sprintf('%02d',round(thrHigh*100));
                                            thrLow = thrHigh*0.4;
                                            w1String=sprintf('%02d',round(w1*100));
                                            w2String=sprintf('%02d',round(w2*100));
                                            w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                            bnImage(:,:,1)=floodHysteresis(ft2(:,:,1),thrHigh,thrLow);
                                            bnImage(:,:,2)=floodHysteresis(ft2(:,:,2),thrHigh,thrLow);
                                            bnImage(:,:,3)=floodHysteresis(ft2(:,:,3),thrHigh,thrLow);
                                            bnImageMaxAgreg(:,:)=max(bnImage,[],3);
                                            bnFileName=strcat(Experiment.bnPrefix,createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                                            bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                            imwrite(bnImageMaxAgreg,strcat(bnFilePath,bnFileNameComp));
                                        end                                   
                                end
                                    
                                clear ft2;
                                clear bnImage;
                                clear bnImageMaxAgreg;
                                clear maxMap;
                            end

                            fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);
                        end

                elseif (strcmp(methodName,'FuzzNYager'))                 
                    
                        if (size(smooImg,3)==3) %If the image is a RGB image do...
                           smooImgHSV=rgb2hsv(smooImg);
                        end

                        for w1 = 0:0.1:1

                            restow1 = 1-w1;

                            for w2 = 0:0.1:restow1
                                    
                                for channel=1:3

                                    [ft,fx,fy]=sobel(smooImgHSV(:,:,channel));
                                    fx=fx./max(ft(:));
                                    fy=fy./max(ft(:));
                                    ft=ft./max(ft(:));
                                    maxMap=directionalNMS(fx,fy);
                                    ft2(:,:,channel)=ft.*maxMap;
                                    clear maxMap ft fx fy;
                                end
                            
                                for idthr=1:length(Experiment.sobel.gray)

                                    thr=Experiment.sobel.gray(idthr);
                                    thrString=sprintf('%02d',round(thr*100));
                                    w1String=sprintf('%02d',round(w1*100));
                                    w2String=sprintf('%02d',round(w2*100));
                                    w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                    ft2Yager=HSVYager(ft2,w1,w2,1-(w1+w2));
                                    bnImage(:,:)=imbinarize(ft2Yager,thr);
                                    bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                    imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                                end
                                     
                                clear ft2;
                                clear ft2Yager;
                                clear bnImage;
                                clear maxMap;
                            end
                        end

                        fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                elseif (strcmp(methodName,'FuzzCanYager'))                 
                    
                        if (size(smooImg,3)==3) %If the image is a RGB image do...
                           smooImgHSV=rgb2hsv(smooImg);
                        end

                        for idSigmaTwo=1:length(Experiment.cannySigmasTwo)

                            sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                            sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-');

                            for w1 = 0:0.1:1

                                restow1 = 1-w1;

                                for w2 = 0:0.1:restow1                             

                                    for channel=1:3

                                        [ft,fx,fy]=canny(smooImgHSV(:,:,channel),sigmaTwo);
                                        fx=fx./max(ft(:));
                                        fy=fy./max(ft(:));
                                        ft=ft./max(ft(:));
                                        maxMap=directionalNMS(fx,fy);
                                        ft2(:,:,channel)=ft.*maxMap;
                                        clear maxMap ft fx fy;
                                    end
                            
                                    for idthr=1:length(Experiment.canny.gray)

                                        thrHigh=Experiment.canny.gray(idthr);
                                        thrString=sprintf('%02d',round(thrHigh*100));
                                        thrLow = thrHigh*0.4;
                                        w1String=sprintf('%02d',round(w1*100));
                                        w2String=sprintf('%02d',round(w2*100));
                                        w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                        ft2Yager=HSVYager(ft2,w1,w2,1-(w1+w2));
                                        bnImage=floodHysteresis(ft2Yager,thrHigh,thrLow);
                                        bnFileName=strcat(Experiment.bnPrefix,createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                                        bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                        imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                                    end
                                    
                                    clear ft2;
                                    clear ft2Yager;
                                    clear bnImage;
                                    clear maxMap;
                                end
                            end 
                            
                            fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);                        
                        
                        end

                elseif (strcmp(methodName,'FuzzDirNYager'))                                
                    
                        if (size(smooImg,3)==3) %If the image is a RGB image do...
                           smooImgHSV=rgb2hsv(smooImg);
                        end

                        for w1 = 0:0.1:1

                            restow1 = 1-w1;

                            for w2 = 0:0.1:restow1
                                                                 
                                    [fx,fy]=sobel(smooImgHSV);
                                    fx2Yager=HSVYager(fx,w1,w2,1-(w1+w2));
                                    fy2Yager=HSVYager(fy,w1,w2,1-(w1+w2));                                     
                                    ftYager=sqrt(fx2Yager.^2+fy2Yager.^2);
                                    fx2Yager=fx2Yager./max(ftYager(:));
                                    fy2Yager=fy2Yager./max(ftYager(:));
                                    maxMap=directionalNMS(fx2Yager,fy2Yager);
                                    ftYager=ftYager./max(ftYager(:));
                                    ft2Yager=ftYager.*maxMap;
                                    
                                    for idthr=1:length(Experiment.sobel.gray)

                                        thr=Experiment.sobel.gray(idthr);
                                        thrString=sprintf('%02d',round(thr*100));
                                        w1String=sprintf('%02d',round(w1*100));
                                        w2String=sprintf('%02d',round(w2*100));
                                        w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                        bnImage(:,:)=imbinarize(ft2Yager,thr);
                                        bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                        imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                                    end
                                    
                                    clear fx2Yager;
                                    clear fy2Yager;
                                    clear ftYager;
                                    clear ft2Yager;
                                    clear bnImage;
                                    clear maxMap;
                            end
                        end

                        fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                elseif (strcmp(methodName,'FuzzDirCYager'))                                
                    
                        if (size(smooImg,3)==3) %If the image is a RGB image do...
                           smooImgHSV=rgb2hsv(smooImg);
                        end
                        
                        for idSigmaTwo=1:length(Experiment.cannySigmasTwo)

                            sigmaTwo=Experiment.cannySigmasTwo(idSigmaTwo);
                            sigmaTwoName=regexprep(sprintf('%.2f',sigmaTwo),'\.','-'); 

                            for w1 = 0:0.1:1

                                restow1 = 1-w1;

                                for w2 = 0:0.1:restow1    
                                                                 
                                    [fx,fy]=canny(smooImgHSV,sigmaTwo);
                                    fx2Yager=HSVYager(fx,w1,w2,1-(w1+w2));
                                    fy2Yager=HSVYager(fy,w1,w2,1-(w1+w2));                                     
                                    ftYager=sqrt(fx2Yager.^2+fy2Yager.^2);
                                    fx2Yager=fx2Yager./max(ftYager(:));
                                    fy2Yager=fy2Yager./max(ftYager(:));
                                    maxMap=directionalNMS(fx2Yager,fy2Yager);
                                    ftYager=ftYager./max(ftYager(:));
                                    ft2Yager=ftYager.*maxMap;
                                    
                                    for idthr=1:length(Experiment.canny.gray)

                                        thrHigh=Experiment.canny.gray(idthr);
                                        thrLow = thrHigh*0.4;
                                        thrString=sprintf('%02d',round(thrHigh*100));
                                        w1String=sprintf('%02d',round(w1*100));
                                        w2String=sprintf('%02d',round(w2*100));
                                        w3String=sprintf('%02d',round((1-(w1+w2))*100));
                                        bnImage(:,:) = floodHysteresis(ft2Yager,thrHigh,thrLow);
                                        bnFileName=strcat(Experiment.bnPrefix,createNameCanny(rawImageName,sigmaOneName,sigmaTwoName,methodName));
                                        bnFileNameComp=strcat(bnFileName,num2str(thrString),'-',num2str(w1String),'-',num2str(w2String),'-',num2str(w3String),'.',Experiment.imagesExt);
                                        imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                                    end
                                    
                                    clear fx2Yager;
                                    clear fy2Yager;
                                    clear ftYager;
                                    clear ft2Yager;
                                    clear bnImage;
                                    clear maxMap;
                                end
                            end

                            fprintf('\t Done for sigmaOne=%.2f, Done for sigmaTwo=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,sigmaTwo,methodName,toc);
                        end

                elseif  (contains(methodName,'SobDifThr&Max'))
                            
                            [ft,fx,fy]=sobel(smooImg);
                            fx=fx./max(ft(:));
                            fy=fy./max(ft(:));
                            ft=ft./max(ft(:));
                            for channel=1:3
                                maxMap(:,:,channel)=directionalNMS(fx(:,:,channel),fy(:,:,channel));
                                ft2(:,:,channel)=ft(:,:,channel).*maxMap(:,:,channel);    
                            end
                            bnImage = zeros(size(ft2));
                            bnImageMax(:,:) = bnImage(:,:,1);
                            for idthrR=1:length(Experiment.sobel.red)
                                thrR=Experiment.sobel.red(idthrR);                                 
                                thrRString=sprintf('%02d',round(thrR*100));                                               
                                for idthrG=1:length(Experiment.sobel.green)
                                    thrG=Experiment.sobel.green(idthrG);                                 
                                    thrGString=sprintf('%02d',round(thrG*100));                                  
                                    for idthrB=1:length(Experiment.sobel.blue)
                                        thrB=Experiment.sobel.blue(idthrB);                                 
                                        thrBString=sprintf('%02d',round(thrB*100));                                 
                                        bnImage(:,:,1)=imbinarize(ft2(:,:,1),thrR);
                                        bnImage(:,:,2)=imbinarize(ft2(:,:,2),thrG);
                                        bnImage(:,:,3)=imbinarize(ft2(:,:,3),thrB);
                                        bnImageMax(:,:)=max(bnImage,[],3);
                                        bnFileNameComp=strcat(bnFileName,'-',num2str(thrRString),'-',num2str(thrGString),'-',num2str(thrBString),'.',Experiment.imagesExt);
                                        imwrite(bnImageMax,strcat(bnFilePath,bnFileNameComp));                                 
                                    end
                                end
                            end

                            clear bnImageMax;
                            clear maxMap;

                            fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);
                            
                elseif  (contains(methodName,'SobDifT&FuMax'))
                            
                            [ft,fx,fy]=sobel(smooImg);
                            fx=fx./max(ft(:));
                            fy=fy./max(ft(:));
                            ft=ft./max(ft(:));
                            for channel=1:3
                                maxMap(:,:,channel)=directionalNMS(fx(:,:,channel),fy(:,:,channel));
                                ft2(:,:,channel)=ft(:,:,channel).*maxMap(:,:,channel);    
                            end
                            ft2max=max(ft2,[],3);
                            bnImage = zeros(size(ft));
                            for idthrR=1:length(Experiment.sobel.red)
                                thrR=Experiment.sobel.red(idthrR);                                 
                                thrRString=sprintf('%02d',round(thrR*100));                                               
                                for idthrG=1:length(Experiment.sobel.green)
                                    thrG=Experiment.sobel.green(idthrG);                                 
                                    thrGString=sprintf('%02d',round(thrG*100));                                  
                                    for idthrB=1:length(Experiment.sobel.blue)
                                        thrB=Experiment.sobel.blue(idthrB);                                 
                                        thrBString=sprintf('%02d',round(thrB*100));
                                        thrVector=[thrR,thrG,thrB];
                                        thrmin=min(thrVector);
                                        bnImage=im2bw(ft2max,thrmin);                                 
                                        bnFileNameComp=strcat(bnFileName,'-',num2str(thrRString),'-',num2str(thrGString),'-',num2str(thrBString),'.',Experiment.imagesExt);
                                        imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));                                 
                                    end
                                end
                            end

                            clear bnImage;
                            clear maxMap;

                            fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);            

                elseif (strcmp(methodName,'SobMeanAgreg'))                    
                        
                        for channel=1:3
                            [ft,fx,fy]=sobel(smooImg(:,:,channel));
                            fx=fx./max(ft(:));
                            fy=fy./max(ft(:));
                            ft=ft./max(ft(:));
                            maxMap=directionalNMS(fx,fy);
                            ft2(:,:,channel)=ft.*maxMap;
                        end
                                                                   
                        for idthr=1:length(Experiment.sobel.gray)
                            thr=Experiment.sobel.gray(idthr);
                            thrString=sprintf('%02d',round(thr*100));
                            ft2mean=mean(ft2,3);
                            bnImage=im2bw(ft2mean,thr);
                            bnFileNameComp=strcat(bnFileName,num2str(thrString),'.',Experiment.imagesExt);
                            imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                        end

                        clear ft2;

                        fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                elseif (strcmp(methodName,'SobMeanDirAgreg'))                    
                                                 
                        [fx,fy]=sobel(smooImg);
                        fxmean=mean(fx,3);
                        fymean=mean(fx,3);
                        ftmean=sqrt(fxmean.^2 + fymean.^2);
                        fxmean=fxmean./max(ftmean(:));
                        fymean=fymean./max(ftmean(:));
                        maxMap=directionalNMS(fxmean,fymean);
                        ftmean=ftmean./max(ftmean(:));
                        ft2mean=ftmean.*maxMap;
                                                                                       
                        for idthr=1:length(Experiment.sobel.gray)

                            thr=Experiment.sobel.gray(idthr);
                            thrString=sprintf('%02d',round(thr*100));
                            bnImage=im2bw(ft2mean,thr);
                            bnFileNameComp=strcat(bnFileName,num2str(thrString),'.',Experiment.imagesExt);
                            imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                        end

                        clear fx;
                        clear fy;
                        clear ft2mean;
                        clear maxMap;

                    fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                elseif (strcmp(methodName,'SobMaxDirAgreg'))                    
                                                 
                        [fx,fy]=sobel(smooImg);
                        fxmax=max(fx,[],3);
                        fymax=max(fx,[],3);
                        ftmax=sqrt(fxmax.^2 + fymax.^2);
                        fxmax=fxmax./max(ftmax(:));
                        fymax=fymax./max(ftmax(:));
                        maxMap=directionalNMS(fxmax,fymax);
                        ftmax=ftmax./max(ftmax(:));
                        ft2max=ftmax.*maxMap;
                                                                                       
                        for idthr=1:length(Experiment.sobel.gray)
                            thr=Experiment.sobel.gray(idthr);
                            thrString=sprintf('%02d',round(thr*100));
                            bnImage=imbinarize(ft2max,thr);
                            bnFileNameComp=strcat(bnFileName,num2str(thrString),'.',Experiment.imagesExt);
                            imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                        end

                        clear ft2max;
                        clear maxMap;

                        fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);

                elseif (strcmp(methodName,'SobMaxAgreg'))                    
                        
                        [ft,fx,fy]=sobel(smooImg);
                        
                        for channel=1:3

                            fx(:,:,channel)=fx(:,:,channel)./max(ft(:,:,channel));
                            fy(:,:,channel)=fy(:,:,channel)./max(ft(:,:,channel));
                            ft(:,:,channel)=ft(:,:,channel)./max(ft(:,:,channel));
                            maxMap=directionalNMS(fx(:,:,channel),fy(:,:,channel));
                            ft2(:,:,channel)=ft(:,:,channel).*maxMap;
                        end
                                                                   
                        for idthr=1:length(Experiment.sobel.gray)

                            thr=Experiment.sobel.gray(idthr);
                            thrString=sprintf('%02d',round(thr*100));
                            ft2max=max(ft2,[],3);
                            bnImage=imbinarize(ft2max,thr);
                            bnFileNameComp=strcat(bnFileName,num2str(thrString),'.',Experiment.imagesExt);
                            imwrite(bnImage,strcat(bnFilePath,bnFileNameComp));
                        end
                        
                        fprintf('\t Done for sigmaOne=%.2f ,binarization=%s (%.2f secs)\n',sigmaOne,methodName,toc);
                              
                end
            end
        end            
        clear('ImgHSVYager','smooImg','smooImgGrey','smooImgHSV','fx','fy','ft','ft2');
    end
end