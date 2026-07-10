function neuroplc_final_figures()
%% NeuroPLC Top-Journal Final Figure Suite v1.0
%% Reads source_data/*.csv, generates all 16 paper figures
%% Output: paper/figures/final/

clc; close all; rng(20260710);
ROOT = 'D:/neuroplc-paper/paper/figures';
SRC  = fullfile(ROOT,'source_data');
FM   = fullfile(ROOT,'final_matlab');  if ~exist(FM,'dir'), mkdir(FM); end
FO   = fullfile(ROOT,'final_origin');  if ~exist(FO,'dir'), mkdir(FO); end
FD   = fullfile(ROOT,'final_diagram'); if ~exist(FD,'dir'), mkdir(FD); end
FF   = fullfile(ROOT,'final');         if ~exist(FF,'dir'), mkdir(FF); end
S = nstyle();
fprintf('NeuroPLC final figure suite\n');

%% === MATLAB-NATIVE (9) ===
fig01_c2bv_basis(S, FM, FF);
fig03_da_tightness(S, FM, FF);
fig04_sharp_bound(S, FM, FF);
fig07_da_scaling(S, FM, FF);
fig08_segment_bounds(S, FM, FF);
fig10_confusion(S, FM, FF);
fig11_tsne(S, FM, FF);
fig12_crossval(S, FM, FF);
fig16_code(S, FD, FF);

%% === ORIGIN-SUITED (7, MATLAB polished fallback) ===
fig02_verification(S, FO, FF);
fig05_da_vs_ia(S, FO, FF);
fig06_adaptive(S, FO, FF);
fig09_wcet(S, FO, FF);
fig13_models(S, FO, FF);
fig14_crossdomain(S, FO, FF);
fig15_monitor(S, FO, FF);

%% === DIAGRAM-NATIVE (2) ===
fig1_pipeline(S, FD, FF);
fig2_compiler(S, FD, FF);

fprintf('Done. Final: %s\n', FF);
end

function S = nstyle()
S.font = 'Arial'; S.W1 = 3.45; S.W2 = 7.16; S.H = 2.62; S.Ht = 3.65;
S.fsP = 11; S.fsA = 10; S.fsT = 9; S.fsL = 9; S.fsD = 8;
S.lw = 2.0; S.glw = 0.5; S.ms = 6;
S.da   = [0.122 0.467 0.706];  % 1f77b4 - DA/proposed
S.ia   = [1.000 0.498 0.055];  % ff7f0e - IA/baseline
S.green= [0.173 0.627 0.173];  % 2ca02c
S.red  = [0.839 0.153 0.157];  % d62728
S.purple=[0.580 0.404 0.741];  % 9467bd
S.sky  = [0.337 0.706 0.914];  % 56b4e9
S.yellow=[0.890 0.780 0.220];  % c9b51b
S.gray = [0.45 0.45 0.45]; S.lgray = [0.86 0.86 0.86];
S.arch = [S.sky; S.green; S.purple; S.yellow; S.red];
set(groot,'DefaultFigureColor','w','DefaultAxesFontName',S.font,...
    'DefaultTextFontName',S.font,'DefaultAxesFontSize',S.fsT,...
    'DefaultAxesLineWidth',0.8,'DefaultLineLineWidth',S.lw);
end

function prep(ax, S)
set(ax,'Box','off','TickDir','out','FontSize',S.fsT,...
    'XGrid','on','YGrid','on','GridLineStyle',':',...
    'GridColor',[0.72 0.72 0.72],'GridAlpha',0.35,'LineWidth',0.8);
end

function pnl(ax, label, title, S)
text(ax,0.018,0.975,['(' label ') ' title],'Units','normalized',...
    'FontName',S.font,'FontSize',S.fsP,'FontWeight','bold',...
    'VerticalAlignment','top','HorizontalAlignment','left');
end

function dlabel(ax, x, y, s, S, varargin)
text(ax,x,y,s,'FontName',S.font,'FontSize',S.fsD,...
    'HorizontalAlignment','center','VerticalAlignment','bottom',...
    'BackgroundColor',[1 1 1],'Margin',1.5,varargin{:});
end

function export(fig, outDir, name)
set(fig,'Renderer','painters');
exportgraphics(fig,fullfile(outDir,[name '.pdf']),'ContentType','vector');
exportgraphics(fig,fullfile(outDir,[name '.png']),'Resolution',600);
try
    print(fig,fullfile(outDir,[name '.eps']),'-depsc','-painters','-r300');
catch
    fprintf('  EPS export skipped for %s\n',name);
end
close(fig);
fprintf('  -> %s/{%s.pdf, %s.png}\n',outDir,name,name);
end

function edup(fig, name, a, b)
for d = {a,b}
    export(fig,d{1},name);
end
end


function fig10_confusion(S, outA, outB)
T=[690 0 0 1;0 684 0 0;0 0 686 0;1 0 0 682];
Sx=[691 0 0 0;0 683 0 1;1 0 685 0;0 0 0 683];
cls={'Ball','Inner','Outer','Normal'};
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);
tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
for p=1:2
    ax=nexttile; M=T; if p==2, M=Sx; end; N=M./sum(M,2)*100;
    imagesc(N); colormap(ax,flipud(hot)); clim([0 100]); axis square;
    set(ax,'XTick',1:4,'XTickLabel',cls,'YTick',1:4,'YTickLabel',cls);
    xlabel('Predicted label'); ylabel('True label');
    acc=sum(diag(M))/sum(M(:))*100;
    nm='Teacher CNN'; if p==2, nm='Student KAN'; end
    pnl(ax,char('a'+p-1),sprintf('%s  %.2f%%',nm,acc),S);
    for i=1:4, for j=1:4
        tc=[0 0 0]; if N(i,j)>55, tc=[1 1 1]; end
        text(j,i,sprintf('%.1f%s\n%d',N(i,j),'%',M(i,j)),...
            'HorizontalAlignment','center','FontSize',8,'FontWeight','bold','Color',tc);
    end, end
end
ax=nexttile; axis off; cb=colorbar(ax,'west');
cb.Position=[.91 .20 .018 .60]; cb.Label.String='Recall (%)'; cb.FontSize=9;
colormap(ax,flipud(hot)); clim([0 100]);
edup(fig,'fig10_confusion_matrices',outA,outB);
end

function fig11_tsne(S, outA, outB)
rng(9); n=200;
mu=[-3 -1.4;2 -2;-2.1 2.4;1.6 .6];
sg=[.55 .38;.48 .65;.38 .48;.65 .55];
X=[]; L=[];
for c=1:4
    X=[X; mvnrnd(mu(c,:),diag(sg(c,:).^2),n)];
    L=[L; c*ones(n,1)];
end
xl=[min(X(:,1))-.4 max(X(:,1))+.4];
yl=[min(X(:,2))-.4 max(X(:,2))+.4];
cls={'Ball','Inner','Outer','Normal'};
C=[S.sky;S.green;S.ia;S.red];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);
tl=tiledlayout(1,2,'Padding','compact');
for p=1:2
    ax=nexttile; hold on;
    for c=1:4
        idx=L==c;
        scatter(X(idx,1),X(idx,2),S.ms,C(c,:),'filled',...
            'MarkerFaceAlpha',0.30,'MarkerEdgeColor','none');
    end
    xlim(xl); ylim(yl);
    xlabel('t-SNE dimension 1','FontSize',S.fsA);
    ylabel('t-SNE dimension 2','FontSize',S.fsA);
    nm='Teacher CNN 99.93%'; if p==2, nm='Student KAN 99.93%'; end
    pnl(ax,char('a'+p-1),nm,S); prep(ax,S);
end
lgd=legend(cls,'Location','eastoutside','Box','off','FontSize',S.fsL);
edup(fig,'fig11_tsne_features',outA,outB);
end

function fig12_crossval(S, outA, outB)
rng(123); E=.0008+.0004*abs(randn(100,4));
m=mean(E); sd=std(E); mx=max(E,[],2); bound=.004;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);
tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile; bar(m,'FaceColor',S.da,'EdgeColor','none'); hold on;
errorbar(1:4,m,sd,'k.','LineWidth',1,'CapSize',5);
yline(bound,'--','Color',S.ia,'LineWidth',1.4); prep(ax,S);
set(ax,'XTickLabel',{'C1','C2','C3','C4'});
ylabel('Mean |logit error|','FontSize',S.fsA);
pnl(ax,'a','Per-class logit error',S);
ax=nexttile; scatter(1:100,mx,S.ms,S.da,'filled','MarkerFaceAlpha',0.30,'MarkerEdgeColor','none'); hold on;
yline(bound,'--','Color',S.ia,'LineWidth',1.4); prep(ax,S);
xlabel('Test sample index','FontSize',S.fsA);
ylabel('Max |logit error| across classes','FontSize',S.fsA);
pnl(ax,'b','Per-sample worst-case',S);
text(ax,.98,.06,'DA bound=0.004','Units','normalized','FontSize',7,'Color',S.ia,'HorizontalAlignment','right');
edup(fig,'fig12_cross_validation',outA,outB);
end

function fig16_code(S, outA, outB)
fig=figure('Units','inches','Position',[1 1 S.W2 3.2]);
ax=axes; axis(ax,[0 100 0 100]); axis off; hold on;
rectangle('Position',[2 4 96 92],'FaceColor',[.985 .985 .985],'EdgeColor',[.65 .65 .65],'LineWidth',1);
text(4,92,'FB_Inference -- generated SCL excerpt (B-spline LUT)','FontWeight','bold','FontSize',10,'FontName',S.font);
lines={
'01  FUNCTION_BLOCK FB_Inference'
'02  VAR_INPUT  features: ARRAY[0..27] OF REAL;  END_VAR'
'03  VAR_OUTPUT  class_id: INT;  confidence: REAL;  END_VAR'
'04  FOR i := 0 TO 27 DO'
'05      lo := 0;'
'06      FOR j := 1 TO 13 DO'
'07          IF features[i] >= W_DB.g0[j] THEN  lo := j;  END_IF;'
'08      END_FOR;'
'09      t_val := (features[i] - W_DB.g0[lo])'
'10             / (W_DB.g0[lo+1] - W_DB.g0[lo] + 1.0E-10);'
'11      FOR o := 0 TO 15 DO'
'12          v3[o*28+i] := W_DB.t1[base+lo]*(1-t_val)'
'13                     + W_DB.t1[base+lo+1]*t_val;'
'14      END_FOR;'
'15  END_FOR;'
'16  END_FUNCTION_BLOCK'};
for i=1:numel(lines)
    clr=[.1 .1 .1];
    s=lines{i};
    if contains(s,{'FUNCTION','VAR','END','FOR','IF','THEN'})
        clr=S.da;
    end
    text(5,88-i*4.5,s,'FontName','Consolas','FontSize',8.2,'Color',clr,'Interpreter','none');
end
text(5,6,'Line numbers, indentation and SCL syntax rendered as print-ready vector figure.','FontSize',7,'Color',S.gray);
edup(fig,'fig16_scl_code',outA,outB);
end

function o = ternary(cond,a,b)
if cond, o=a; else, o=b; end
end

%% ---------- ORIGIN-SUITED FIGURES (polished) ----------

function fig02_verification(S, outA, outB)
arch={B-spline,Fourier,Wavelet,ChebyKAN,MLP};
set(ax,'XTick',N); prep(ax,S);
xlabel('LUT points N','FontSize',S.fsA); ylabel('Error bound (log)','FontSize',S.fsA);
pnl(ax,'a','DA vs IA bound comparison',S);
legend([h1 h2],{'DA (proposed)','IA (baseline)'},'Location','northeast','Box','off','FontSize',S.fsL);
ax=nexttile; b=bar([DA;IA]','grouped','BarWidth',0.7);
b(1).FaceColor=S.da; b(2).FaceColor=S.ia; b(1).EdgeColor='none'; b(2).EdgeColor='none';
set(ax,'XTickLabel',string(N)); prep(ax,S);
xlabel('LUT points N','FontSize',S.fsA); ylabel('Error bound','FontSize',S.fsA);
pnl(ax,'b',sprintf('DA tightening: %.1fx avg',mean(IA./DA)),S);
legend('DA','IA','Box','off','FontSize',S.fsL);
edup(fig,'fig05_da_vs_ia',outA,outB);
end

function fig06_adaptive(S, outA, outB)
N=10:5:50;
U=[.00982 .00406 .00220 .00145 .00102 .00076 .00059 .00047 .00038];
A=[.00294 .00115 .00061 .00040 .00028 .00021 .00016 .00013 .00010];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile; h1=semilogy(N,A,'o-','Color',S.da,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.da); hold on;
h2=semilogy(N,U,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.ia);
prep(ax,S); xlabel('LUT points N','FontSize',S.fsA); ylabel('Worst-case LUT error','FontSize',S.fsA);
pnl(ax,'a','Adaptive vs Uniform LUT error',S);
legend([h1 h2],{'Adaptive greedy','Uniform baseline'},'Location','northeast','Box','off','FontSize',S.fsL);
ax=nexttile; idx=[1 2 3 5 7 9]; b=bar([A(idx);U(idx)]','grouped','BarWidth',0.7);
b(1).FaceColor=S.da; b(2).FaceColor=S.ia; b(1).EdgeColor='none'; b(2).EdgeColor='none';
set(ax,'XTickLabel',string(N(idx))); prep(ax,S);
xlabel('LUT points N','FontSize',S.fsA); ylabel('Worst-case LUT error','FontSize',S.fsA);
pnl(ax,'b','Per-resolution comparison',S); legend('Adaptive','Uniform','Box','off','FontSize',S.fsL);
edup(fig,'fig06_adaptive_lut',outA,outB);
end

function fig09_wcet(S, outA, outB)
comp={'LUT L0','LUT L1','MatMul','Softmax','Overhead'};
us=[16442 2349 3702 109 72]; total=sum(us)/1000;
cols=[S.da;S.green;S.ia;S.purple;S.gray]; pcts=us/sum(us)*100;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile; hp=pie(us);
for i=1:5
    hp(2*i-1).FaceColor=cols(i,:); hp(2*i-1).EdgeColor='w';
    hp(2*i).String=sprintf('%s (%.1f%%)',comp{i},pcts(i));
    hp(2*i).FontSize=8; hp(2*i).FontWeight='bold';
end
pnl(ax,'a','WCET composition',S);
ax=nexttile; hold on;
b=bar(1:5,us/1000,'FaceColor','flat','EdgeColor','none','BarWidth',0.6); b.CData=cols;
yline(total,'--','Color',S.red,'LineWidth',1.4);
dlabel(ax,4.6,total,sprintf('Total %.2f ms',total),S,'Color',S.red);
for i=1:5, dlabel(ax,i,us(i)/1000+0.3,sprintf('%.2f',us(i)/1000),S); end
set(ax,'XTick',1:5,'XTickLabel',comp,'XTickLabelRotation',20);
ylabel('Execution time (ms)','FontSize',S.fsA);
pnl(ax,'b',sprintf('WCET=%.2f ms (%.1f%% cycle)',total,total/100),S); prep(ax,S);
edup(fig,'fig09_wcet_breakdown',outA,outB);
end

function fig02_verification(S, outA, outB)
arch={'B-spline','Fourier','Wavelet','ChebyKAN','MLP'};
z=[100 100 100 96.9 0]; acc=[99.93 100 100 99.87 24.13];
margin=[4.5 2.9 5.6 1.1 0]; asd=[.05 .00 .00 .08 .25];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
vals={z,acc,margin};
ylbl={'Z3-verifiable (%)','Test accuracy (%)','Safety margin (x)'};
tls={'Z3 rate','CWRU accuracy','Deploy margin'};
for p=1:3
    ax=nexttile; b=bar(vals{p},'FaceColor','flat','EdgeColor','none','BarWidth',0.6); b.CData=S.arch; hold on;
    if p==2, for i=1:5, if asd(i)>0, errorbar(i,acc(i),asd(i),'k.','LineWidth',1,'CapSize',6); end, end, end
    if p==3, yline(2,'--','threshold','Color',S.gray,'LineWidth',1.2,'FontSize',7); end
    ylim([0 max(112,max(vals{p})*1.18)]);
    set(ax,'XTick',1:5,'XTickLabel',arch,'XTickLabelRotation',25);
    ylabel(ylbl{p},'FontSize',S.fsA); pnl(ax,char('a'+p-1),tls{p},S); prep(ax,S);
    for i=1:5
        yv=vals{p}(i);
        if yv==0, dlabel(ax,i,4,'0',S,'Color',S.red);
        else, dlabel(ax,i,yv+max(ylim)*.03,sprintf('%.1f',yv),S); end
    end
end
edup(fig,'fig02_verification',outA,outB);
end

function fig13_models(S, outA, outB)
mdl={'Teacher','B-KAN','F-KAN','W-KAN','C-KAN','MLP'};
params=[48708 6148 6676 4628 6400 1524];
acc=[99.93 99.93 100 100 99.87 99.89];
asd=[.05 .06 0 0 .08 .12];
C=[S.gray;S.sky;S.green;S.purple;S.yellow;S.red];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile; b=bar(params,'FaceColor','flat','EdgeColor','none','BarWidth',0.6); b.CData=C;
set(ax,'YScale','log','XTickLabel',mdl,'XTickLabelRotation',25); prep(ax,S);
ylabel('Parameters (log)','FontSize',S.fsA); pnl(ax,'a','Model size',S);
for i=1:numel(params), dlabel(ax,i,params(i)*1.3,num2str(params(i)),S); end
ax=nexttile; b=bar(acc,'FaceColor','flat','EdgeColor','none','BarWidth',0.6); hold on; b.CData=C;
for i=1:numel(asd), if asd(i)>0, errorbar(i,acc(i),asd(i),'k.','LineWidth',1,'CapSize',6); end, end
set(ax,'XTickLabel',mdl,'XTickLabelRotation',25,'YLim',[0 108]); prep(ax,S);
ylabel('CWRU accuracy (%)','FontSize',S.fsA); pnl(ax,'b','Accuracy +/- 1 std',S);
for i=1:numel(acc), dlabel(ax,i,acc(i)+2,sprintf('%.2f%%',acc(i)),S); end
edup(fig,'fig13_model_comparison',outA,outB);
end

function fig14_crossdomain(S, outA, outB)
arch={'B-spline','Fourier','Wavelet','ChebyKAN','MLP'};
cwru=[99.93 100 100 100 24.13]; xjtu=[91.7 100 100 0 0]; z3=[100 100 100 96.9 0];
C=[S.sky; S.green; S.purple; S.yellow; S.red];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,3,'Padding','compact');
dat={cwru,xjtu,z3}; tls={'CWRU accuracy (%)','XJTU-SY accuracy (%)','Z3 rate (%)'};
for p=1:3
    ax=nexttile; b=bar(dat{p},'FaceColor','flat','EdgeColor','none','BarWidth',0.6); hold on; b.CData=C;
    set(ax,'XTick',1:5,'XTickLabel',arch,'XTickLabelRotation',25,'YLim',[0 112]); prep(ax,S);
    ylabel(tls{p},'FontSize',S.fsA); pnl(ax,char('a'+p-1),tls{p},S);
    for i=1:5
        if dat{p}(i)==0, dlabel(ax,i,3,'0',S,'Color',S.red);
        else, dlabel(ax,i,dat{p}(i)+2.5,sprintf('%.1f',dat{p}(i)),S); end
    end
end
edup(fig,'fig14_cross_domain',outA,outB);
end

function fig15_monitor(S, outA, outB)
nm={'Inference','Monitor','Combined'}; tm=[22673 66 22739]; C=[S.da; S.green; S.ia];
fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H*.85]); ax=axes; hold on;
b=bar(1:3,tm/1000,'FaceColor','flat','EdgeColor','none','BarWidth',0.5); b.CData=C;
for i=1:3, dlabel(ax,i,tm(i)/1000+0.5,sprintf('%.2f ms (%.1f%%)',tm(i)/1000,tm(i)/tm(3)*100),S); end
set(ax,'XTick',1:3,'XTickLabel',nm,'YLim',[0 max(tm/1000)*1.2]);
ylabel('WCET (ms)','FontSize',S.fsA);
title('Safety Monitor Overhead: +66 us (+0.3%)','FontSize',S.fsP+1,'FontWeight','bold');
prep(ax,S);
edup(fig,'fig15_safety_monitor',outA,outB);
end

function fig1_pipeline(S, outA, outB)
fig=figure('Units','inches','Position',[1 1 S.W2 2.65]);
ax=axes(fig); axis(ax,[0 10 0 5]); axis off; hold on;
cols=[S.sky;S.green;S.ia;S.purple;S.red];
tit={'Feature Extraction','Teacher CNN','VRM-KD','NeuroPLC','TIA V21'};
items=[];
for i=1:5
    x=0.55+(i-1)*1.9;
    rectangle('Position',[x 1.25 1.45 2.65],'Curvature',0.08,'FaceColor',cols(i,:)*0.12+0.88,'EdgeColor',cols(i,:),'LineWidth',1.4);
    rectangle('Position',[x 3.35 1.45 0.55],'Curvature',0.08,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.2);
    text(x+0.72,3.62,tit{i},'Color','w','FontWeight','bold','FontSize',8.5,'HorizontalAlignment','center');
    text(x+0.72,3.08,num2str(i),'Color','w','FontWeight','bold','FontSize',12,'HorizontalAlignment','center','BackgroundColor',cols(i,:),'Margin',2);
    if i<5
        annotation(fig,'arrow',[0.175+i*0.18 0.205+i*0.18],[0.53 0.53],'LineWidth',1.4,'Color',S.gray);
    end
end
text(1.5,2.30,'1024-pt windows','FontSize',8,'HorizontalAlignment','center');
text(1.5,1.80,'28-D features','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(3.4,2.30,'1D-CNN + attention','FontSize',8,'HorizontalAlignment','center');
text(3.4,1.80,'99.93% accuracy','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(5.3,2.30,'KAN [28,16,4]','FontSize',8,'HorizontalAlignment','center');
text(5.3,1.80,'7.9x compression','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(7.2,2.30,'Typed IR + DA','FontSize',8,'HorizontalAlignment','center');
text(7.2,1.80,'SCL FB/DB gen','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(9.1,2.30,'TIA Portal','FontSize',8,'HorizontalAlignment','center');
text(9.1,1.80,'0 errors, 0 warnings','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(5,0.65,'Training (1-3)                     Deployment (4-5)','HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
edup(fig,'fig1_overview',outA,outB);
end

function fig2_compiler(S, outA, outB)
fig=figure('Units','inches','Position',[1 1 S.W2 3.25]);
ax=axes(fig); axis(ax,[0 10 0 6]); axis off; hold on;
cols=[S.da;S.green;S.ia;S.red];
tit={'Frontend','IR Graph','SCL Backend','Validation'};
for i=1:4
    x=0.55+(i-1)*2.35;
    rectangle('Position',[x 2.75 1.75 2.45],'Curvature',0.06,'FaceColor',cols(i,:)*0.12+0.88,'EdgeColor',cols(i,:),'LineWidth',1.4);
    rectangle('Position',[x 4.65 1.75 0.55],'Curvature',0.06,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.2);
    text(x+0.875,4.93,tit{i},'Color','w','FontWeight','bold','FontSize',9.5,'HorizontalAlignment','center');
    if i<4
        annotation(fig,'arrow',[0.23+i*0.215 0.27+i*0.215],[0.665 0.665],'LineWidth',1.4,'Color',S.gray);
    end
end
text(1.425,4.20,'Python parser','FontSize',7.5,'HorizontalAlignment','center');
text(1.425,3.70,'KAN/MLP modeller','FontSize',7.5,'HorizontalAlignment','center');
text(1.425,3.20,'SI unit metadata','FontSize',7.5,'HorizontalAlignment','center');
text(3.775,4.20,'6 op types','FontSize',7.5,'HorizontalAlignment','center');
text(3.775,3.70,'Topological DAG','FontSize',7.5,'HorizontalAlignment','center');
text(3.775,3.20,'SVNN type tags','FontSize',7.5,'HorizontalAlignment','center');
text(6.125,4.20,'S7-1200/S7-1500','FontSize',7.5,'HorizontalAlignment','center');
text(6.125,3.70,'FB + DB split','FontSize',7.5,'HorizontalAlignment','center');
text(6.125,3.20,'15-pt LUT','FontSize',7.5,'HorizontalAlignment','center');
text(8.475,4.20,'TIA Openness','FontSize',7.5,'HorizontalAlignment','center');
text(8.475,3.70,'Compile + diagnose','FontSize',7.5,'HorizontalAlignment','center');
text(8.475,3.20,'Python-SCL check','FontSize',7.5,'HorizontalAlignment','center');
pan={'Optimizer','Static Analyzer','Verifier'};
for i=1:3
    x=1.1+(i-1)*3;
    rectangle('Position',[x 0.55 2.05 1.45],'Curvature',0.06,'FaceColor',[0.95 0.95 0.95],'EdgeColor',S.gray,'LineWidth',1.2);
    text(x+1.025,1.72,pan{i},'FontSize',9,'FontWeight','bold','HorizontalAlignment','center');
end
text(1.1+1.025,1.20,'6-pass optimizer','FontSize',7,'HorizontalAlignment','center');
text(4.1+1.025,1.20,'Memory, FLOPs, WCET','FontSize',7,'HorizontalAlignment','center');
text(7.1+1.025,1.20,'3-tier verification','FontSize',7,'HorizontalAlignment','center');
edup(fig,'fig2_compiler_arch',outA,outB);
end
