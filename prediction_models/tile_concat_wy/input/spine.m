function [mask,counter2,locations]=slide_analysis2(im,scale,patch_size,step,x_ind1,x_ind2)
%% input images, scale, patch_size, step
if(nargin<2)
    scale=1;
end
if(nargin<3)
    patch_size=512;
end
if(nargin<4)
    step=patch_size;
end
if(nargin<5) %% what is x_ind1 and x_ind2
    x_ind1=1;
end
if(nargin<6)
    x_ind2=size(im,2);
end

colors={'yellow','red','green','blue','black'};

patch_width=patch_size/scale;
box_width=patch_width;
patch_height=patch_size/scale;
box_height=patch_height;
step_size=step/scale;
step_size2=step/scale;


rad=5; %strel radius in pixels
thresh=.9; %threshold for mask excluding the white space
min_size=40; %% smallest componets size
min_size2=min_size*scale;
tan_size=4;
bthresh=20;
hsvthresh=.1; % water stains and bubbles are removed by setting a lower bound on saturation which is 0.1
slide_thresh=.6; % patches that intersect with less than 60% intersection with the mask are removed
im2=im;
im3=mean(im2,3); %% returns the mean over third dimension
im4=im3<thresh*max(im3(:));
im4(im2(:,:,2)>im3)=0;
im4(im2(:,:,3)>im2(:,:,1)+bthresh)=0; %% what is this about
im2hsv=rgb2hsv(im2);
cc=bwconncomp(im4); % bwconncomp: Find connected components in binary image
im5=im4;
for i=1:length(cc.PixelIdxList)
    if length(cc.PixelIdxList{i})<min_size
        im5(cc.PixelIdxList{i})=0;
    end
end

### Not implementing
S=createSphericalStrel(rad); % morphological smoothing object
im6=convn(im5,S,'same')>0; % convolution im5 with S
im6(:,[1,end])=0; %% why is these two lines
im6([1,end],:)=0;
im7=1-(convn(1-im6,S,'same')>0);

im8=imfill(im7); % fill image regions and wholes
im8=bwmorph(im8,'skel',Inf); % morphological operations on binary images, find skelonton
im8 = bwmorph(im8,'thin',Inf); % prunning

cc=bwconncomp(im8);  % bwconncomp: Find connected components in binary image
im9=zeros(size(im8)); % or zeros with the same size
for i=1:length(cc.PixelIdxList)
    tim=zeros(size(im8));
    tim(cc.PixelIdxList{i})=1;
    E=find(bwmorph(tim,'endpoints')); % Find indices and values of nonzero elements
    maxind1=-1;
    maxind2=-1;
    maxval=-1;
    for j=1:length(E) %% find the longest endpoints in the image
        tim2=bwdistgeodesic(im8,E(j)); % compute the geodesic distance with each endpoints pair
        tim2(isinf(tim2))=0;
        if(max(tim2(:))>maxval)
            maxind1=E(j);
            maxval=max(tim2(:));
            maxind2=find(tim2==maxval);
        end
    end
    maxind2=maxind2(1);

    E=setdiff(E,[maxind1,maxind2]);
    while numel(E)>0 ## recursively set the longest skelonton
        tim(E)=0;
        E = find(bwmorph(tim, 'endpoints'));
        E=setdiff(E,[maxind1,maxind2]);
    end
    im9=im9+tim;
end
im9 = bwmorph(im9,'thin',Inf);
im9(:,1:x_ind1)=0;
im9(:,x_ind2:end)=0;

mask=im5;
mask(:,1:x_ind1)=0;
mask(:,x_ind2:end)=0;

imshow(im2,[]);
hold on;

cc=bwconncomp(im9);
counter2=0;
patches=[];
locations=[];
masks=[];
for i=1:length(cc.PixelIdxList)
    tim=zeros(size(im9));
    tim(cc.PixelIdxList{i})=1;
    E = bwmorph(tim, 'endpoints');
    [y,x] = find(E);
    pts=[];
    dists=[];
    counter=1;
    pts(counter,:)=[y(1),x(1)];
    tim(pts(counter,1),pts(counter,2))=0;
    while sum(tim(:)>0)
        counter=counter+1;
        [y,x]=find(tim(pts(counter-1,1)+(-1:1),pts(counter-1,2)+(-1:1)));
        dists(counter)=1+(abs(x)==abs(y))*(sqrt(2)-1);
        pts(counter,:)=pts(counter-1,:)+[y,x]-2;
        tim(sub2ind(size(tim),pts(counter,1),pts(counter,2)))=0;
    end %% what does this part do measure the distance?
%     dists
%
    tim=zeros(size(im9));
    rad=box_width/2;
    rad2=box_height/2;
    rad3=box_height*2;
    j=1;
    location=0;
    while(j<(tan_size+1))
        j=j+1;
        location=location+dists(j);
    end
    next_step=location+step_size;
    while(j<=(size(pts,1)-tan_size-1))
        jr=round(j);
        pt=pts(jr,:);
        p1=pts(jr-tan_size,:);
        p2=pts(jr+tan_size,:);
        tan=(p2-p1)/sqrt(sum((p2-p1).^2));

        x=pt(2)+[rad*tan(2)-rad2*tan(1),-rad*tan(2)-rad2*tan(1),-rad*tan(2)+rad2*tan(1),rad*tan(2)+rad2*tan(1)];
        y=pt(1)+[rad*tan(1)+rad2*tan(2),-rad*tan(1)+rad2*tan(2),-rad*tan(1)-rad2*tan(2),rad*tan(1)-rad2*tan(2)];

        x2=pt(2)+[-rad3*tan(1),rad3*tan(1)];
        y2=pt(1)+[rad3*tan(2),-rad3*tan(2)];
        x2=linspace(x2(1),x2(2));
        y2=linspace(y2(1),y2(2));
        y2=y2(and(x2>=1,x2<=size(tim,2)));
        x2=x2(and(x2>=1,x2<=size(tim,2)));
        x2=x2(and(y2>=1,y2<=size(tim,1)));
        y2=y2(and(y2>=1,y2<=size(tim,1)));

        valid=mask(sub2ind(size(im7),round(y2),round(x2)))>0;
        if(sum(valid)<2)
            valid(floor(length(valid)/2)+(1:2))=1;
        end
        x2=x2(valid);
        y2=y2(valid);

        x2=x2([1,end]);
        y2=y2([1,end]);

        dist=sqrt((x2(2)-x2(1))^2+(y2(2)-y2(1))^2);
        n=round((dist-patch_height+step_size2)/(step_size2));
        delta=max(0,dist-n*step_size2-(patch_height-step_size2));

        vec=([x2(2),y2(2)]-[x2(1),y2(1)])/dist;
        x2(1)=x2(1)+vec(1)*delta/2;
        x2(2)=x2(2)-vec(1)*delta/2;
        y2(1)=y2(1)+vec(2)*delta/2;
        y2(2)=y2(2)-vec(2)*delta/2;

        dist=dist-delta;

        tim(sub2ind(size(tim),round(y2),round(x2)))=1;
        vec=([x2(2),y2(2)]-[x2(1),y2(1)])/dist;
        x2(2)=x2(2)-vec(1)*patch_height;
        y2(2)=y2(2)-vec(2)*patch_height;
        tim(sub2ind(size(tim),round(y2),round(x2)))=1;
        x3=linspace(x2(1),x2(2),n);
        y3=linspace(y2(1),y2(2),n);

        for k=1:length(x3)
            x=x3(k)+[rad*tan(2),-rad*tan(2),-rad*tan(2)+2*rad*tan(1),rad*tan(2)+2*rad*tan(1)];
            y=y3(k)+[rad*tan(1),-rad*tan(1),-rad*tan(1)-2*rad*tan(2),rad*tan(1)-2*rad*tan(2)];

            line_color='blue';
            bw=poly2mask(x,y,size(mask,1),size(mask,2));

            if(sum(bw(:).*mask(:))>slide_thresh*sum(bw(:)))
                line(x([1,2]),y([1,2]),'Color',line_color);
                line(x([2,3]),y([2,3]),'Color',line_color);
                line(x([3,4]),y([3,4]),'Color',line_color);
                line(x([4,1]),y([4,1]),'Color',line_color);
                counter2=counter2+1;

                x1=linspace(x(1),x(2),patch_size);
                x2=linspace(x(4),x(3),patch_size);
                y1=linspace(y(1),y(2),patch_size);
                y2=linspace(y(4),y(3),patch_size);

                location2=zeros(patch_size,patch_size,2);
                for m=1:length(x1)
                    x4=round(linspace(x1(m),x2(m),patch_size)*scale);
                    y4=round(linspace(y1(m),y2(m),patch_size)*scale);
                    location2(m,:,1)=x4;
                    location2(m,:,2)=y4;
                end
                locations{counter2}=location2;
            end
        end

        if(j==(size(pts,1)-tan_size-1))
            j=j+1;
            location=location+dists(j);
        end
        while(and(location<next_step,j<=(size(pts,1)-tan_size-1)))
            j=j+1;
            location=location+dists(j);
        end
        next_step=next_step+step_size;
    end % for while in line 136
end % for for in line 106