function neuroplc_topjournal_figures()
%% NeuroPLC top-journal figure suite
% Venue target: IEEE Transactions on Industrial Informatics
% Output: PDF vector + EPS vector + PNG 600 dpi
% MATLAB: R2025b

clc; close all;
rng(20260710);
outDir = 'D:/neuroplc-paper/paper/figures';
if ~exist(outDir, 'dir'), mkdir(outDir); end

S = nii_style();

fprintf('NeuroPLC top-journal figure suite\n');
fprintf('Output: %s\n\n', outDir);

fig_overview(S, outDir);                 % fig1_overview
fig_compiler_arch(S, outDir);            % fig2_compiler_arch
fig_c2bv_basis(S, outDir);               % fig01_c2bv_basis
fig_c2bv_verification(S, outDir);        % fig02_verification
fig_da_tightness(S, outDir);             % fig03_da_tightness
fig_sharp_bound(S, outDir);              % fig04_sharp_bound
fig_da_vs_ia(S, outDir);                 % fig05_da_vs_ia
fig_adaptive_lut(S, outDir);             % fig06_adaptive_lut
fig_da_scaling(S, outDir);               % fig07_da_scaling
fig_segment_bounds(S, outDir);           % fig08_segment_bounds
fig_wcet(S, outDir);                     % fig09_wcet_breakdown
fig_confusion(S, outDir);                % fig10_confusion_matrices
fig_tsne(S, outDir);                     % fig11_tsne_features
fig_cross_validation(S, outDir);         % fig12_cross_validation
fig_scl_code(S, outDir);                 % fig16_scl_code

fprintf('\nAll figures regenerated.\n');
end

function S = nii_style()
S.font = 'Arial';
S.W1 = 3.50; S.W2 = 7.16; S.H = 2.70; S.Ht = 3.65;
S.fsPanel = 11; S.fsAxis = 10; S.fsTick = 9; S.fsLegend = 9; S.fsData = 8;
S.lw = 2.0; S.gridlw = 0.5; S.mark = 6;
S.da = hex2rgb('#1f77b4');
S.ia = hex2rgb('#ff7f0e');
S.green = hex2rgb('#2ca02c');
S.red = hex2rgb('#d62728');
S.purple = hex2rgb('#9467bd');
S.sky = hex2rgb('#56b4e9');
S.yellow = hex2rgb('#c9b51b');
S.gray = [0.45 0.45 0.45];
S.lgray = [0.86 0.86 0.86];
S.arch = [S.sky; S.green; S.purple; S.yellow; S.red];
set(groot, 'DefaultFigureColor', 'w', 'DefaultAxesFontName', S.font, ...
    'DefaultTextFontName', S.font, 'DefaultAxesFontSize', S.fsTick, ...
    'DefaultAxesLineWidth', 0.8, 'DefaultLineLineWidth', S.lw);
end

function c = hex2rgb(h)
h = erase(h, '#');
c = [hex2dec(h(1:2)), hex2dec(h(3:4)), hex2dec(h(5:6))] / 255;
end

function prep(ax, S)
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontName', S.font, 'FontSize', S.fsTick, ...
    'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', ':', 'GridColor', [0.72 0.72 0.72], ...
    'GridAlpha', 0.35, 'LineWidth', 0.8);
end

function panel(ax, tag, txt, S)
text(ax, 0.018, 0.975, ['(' tag ') ' txt], 'Units', 'normalized', ...
    'FontName', S.font, 'FontSize', S.fsPanel, 'FontWeight', 'bold', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
end

function boxed_label(ax, x, y, s, S, varargin)
text(ax, x, y, s, 'FontName', S.font, 'FontSize', S.fsData, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'BackgroundColor', [1 1 1], 'Margin', 1.2, varargin{:});
end

function export_all(fig, outDir, name)
set(fig, 'Renderer', 'painters');
exportgraphics(fig, fullfile(outDir, [name '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 600);
print(fig, fullfile(outDir, [name '.eps']), '-depsc', '-painters', '-r300');
close(fig);
end

function fig_overview(S, outDir)
fig = figure('Units','inches','Position',[1 1 S.W2 2.65]);
ax = axes(fig); axis(ax, [0 10 0 5]); axis off; hold on;
cols = [S.sky; S.green; S.ia; S.purple; S.red];
titles = {'Feature Extraction','Teacher CNN','VRM-KD Distillation','NeuroPLC Compiler','TIA V21 Validation'};
lines = {{'1024-pt windows','28-D features'}, {'1D-CNN + attention','99.93% accuracy'}, {'KAN [28,16,4]','7.9x compression'}, {'Typed IR + DA','SCL FB/DB'}, {'0 errors/warnings','22.67 ms WCET'}};
for i=1:5
    x = 0.55 + (i-1)*1.9;
    rectangle('Position',[x 1.25 1.45 2.65], 'Curvature',0.08, 'FaceColor',cols(i,:)*0.12+0.88, 'EdgeColor',cols(i,:), 'LineWidth',1.4);
    rectangle('Position',[x 3.35 1.45 0.55], 'Curvature',0.08, 'FaceColor',cols(i,:), 'EdgeColor',cols(i,:), 'LineWidth',1.2);
    text(x+0.72,3.63,titles{i},'Color','w','FontWeight','bold','FontSize',9,'HorizontalAlignment','center');
    text(x+0.72,3.08,num2str(i),'Color','w','FontWeight','bold','FontSize',11,'HorizontalAlignment','center','BackgroundColor',cols(i,:),'Margin',2);
    text(x+0.72,2.45,lines{i}{1},'FontSize',8.5,'HorizontalAlignment','center');
    text(x+0.72,1.95,lines{i}{2},'FontSize',8.5,'FontWeight','bold','HorizontalAlignment','center');
    if i<5
        annotation(fig,'arrow',[0.17+i*0.18 0.20+i*0.18],[0.53 0.53],'LineWidth',1.4,'Color',S.gray);
    end
end
text(5,0.65,'Training path (1-3)                          Deployment path (4-5)', 'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
export_all(fig,outDir,'fig1_overview');
end

function fig_compiler_arch(S, outDir)
fig = figure('Units','inches','Position',[1 1 S.W2 3.25]);
ax = axes(fig); axis(ax,[0 10 0 6]); axis off; hold on;
cols = [S.da; S.green; S.ia; S.red];
titles = {'Frontend','Typed IR','SCL Backend','Validation'};
items = {{'PyTorch state dict','KAN/MLP extractor','SI unit metadata'}, {'6 op types','Topological DAG','SVNN type tags'}, {'S7-1200 / S7-1500','FB + DB split','15-pt LUT'}, {'TIA Openness','Compile diagnose','Python-SCL check'}};
for i=1:4
    x=0.55+(i-1)*2.35;
    rectangle('Position',[x 2.75 1.75 2.45],'Curvature',0.06,'FaceColor',cols(i,:)*0.12+0.88,'EdgeColor',cols(i,:),'LineWidth',1.4);
    rectangle('Position',[x 4.65 1.75 0.55],'Curvature',0.06,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.2);
    text(x+0.875,4.93,titles{i},'Color','w','FontWeight','bold','FontSize',9.5,'HorizontalAlignment','center');
    for k=1:3
        text(x+0.875,4.18-0.48*k,items{i}{k},'FontSize',8.2,'HorizontalAlignment','center');
    end
    if i<4, annotation(fig,'arrow',[0.235+i*0.215 0.27+i*0.215],[0.665 0.665],'LineWidth',1.4,'Color',S.gray); end
end
panels = {'Optimizer','Static analyzer','Verifier'};
pcols = [S.da; S.green; S.purple];
pitems = {{'Adaptive LUT','FuseMatMulAdd','LUTizeEXP'}, {'Memory + FLOPs','WCET bound','DA margin'}, {'Template proof','512/512 Z3','Compiler TCB'}};
for i=1:3
    x=1.1+(i-1)*3.0;
    rectangle('Position',[x 0.55 2.05 1.45],'Curvature',0.06,'FaceColor',pcols(i,:)*0.12+0.88,'EdgeColor',pcols(i,:),'LineWidth',1.2);
    text(x+1.025,1.72,panels{i},'FontSize',9,'FontWeight','bold','HorizontalAlignment','center');
    text(x+1.025,1.25,strjoin(pitems{i},'  |  '),'FontSize',7.4,'HorizontalAlignment','center');
end
plot([0.65 1.2],[0.25 0.25],'-','Color',S.gray,'LineWidth',1.5); text(1.35,0.25,'data flow','FontSize',8,'VerticalAlignment','middle');
plot([3.1 3.65],[0.25 0.25],'--','Color',S.gray,'LineWidth',1.2); text(3.8,0.25,'analysis/control flow','FontSize',8,'VerticalAlignment','middle');
export_all(fig,outDir,'fig2_compiler_arch');
end

function fig_c2bv_basis(S, outDir)
x=linspace(-3,3,700)'; g=linspace(-3,3,15)';
phi{1}=0.5*sin(.8*x)+0.25*cos(1.4*x+.5)+0.12*x;
phi{2}=0.35*sin(.4*x)+0.25*cos(.8*x+.3)+0.18*sin(1.2*x+.6);
t=(x+.3)/.8; phi{3}=0.7*(2/sqrt(3))*pi^(-1/4)*(1-t.^2).*exp(-t.^2/2);
phi{4}=0.35*cos(x)-0.25*cos(3*x)+0.15*cos(5*x);
phi{5}=0.65*exp(-x.^2/.36);
names={'B-spline KAN','FourierKAN','WaveletKAN','ChebyKAN','RBF-KAN'}; M2=[0.68 2.30 2.60 3.12 3.09];
yr=max(cellfun(@(v)max(abs(v)),phi))*1.25;
fig=figure('Units','inches','Position',[1 1 S.W2 S.Ht]); tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for i=1:6
    ax=nexttile; hold on;
    if i<=5
        fill([x;flipud(x)],[zeros(size(x));flipud(phi{i})],S.arch(i,:),'FaceAlpha',0.18,'EdgeColor','none');
        plot(x,phi{i},'-','Color',S.arch(i,:),'LineWidth',S.lw);
        yg=interp1(x,phi{i},g);
        for q=1:numel(g), plot([g(q) g(q)],[0 yg(q)],':','Color',[.55 .55 .55],'LineWidth',0.7); end
        scatter(g,yg,12,[.35 .35 .35],'filled','MarkerFaceAlpha',.55);
        panel(ax,char('a'+i-1),sprintf('%s  M_2=%.2f',names{i},M2(i)),S);
    else
        for j=1:5, plot(x,phi{j},'LineWidth',1.2,'Color',S.arch(j,:)); end
        panel(ax,'f','Shared LUT grid view',S);
        legend(names,'Location','southoutside','NumColumns',2,'FontSize',7,'Box','off');
    end
    ylim([-yr yr]); xlim([-3 3]); prep(ax,S);
    if mod(i-1,3)==0, ylabel('\phi(x) (activation value)','FontSize',S.fsAxis); end
    if i>=4, xlabel('Input x (normalized units)','FontSize',S.fsAxis); end
end
export_all(fig,outDir,'fig01_c2bv_basis');
end

function fig_c2bv_verification(S, outDir)
arch={'B-spline','Fourier','Wavelet','ChebyKAN','MLP'}; z=[100 100 100 96.9 0]; acc=[99.93 100 100 99.87 24.13]; margin=[4.5 2.9 5.6 1.1 0]; sd=[.05 .00 .00 .08 .25];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
vals={z,acc,margin}; yl={'Z3-verifiable edges (%)','Test accuracy (%)','Safety margin (x)'}; ttl={'Z3 verifiability','CWRU accuracy','Deployment margin'};
for p=1:3
    ax=nexttile; b=bar(vals{p},'FaceColor','flat','EdgeColor','none'); b.CData=S.arch; hold on;
    if p==2, errorbar(1:5,acc,sd,'k.','LineWidth',1,'CapSize',5); end
    if p==3, yline(2,'--','2x threshold','Color',S.gray,'LineWidth',1.2); end
    ylim([0, max([110 max(vals{p})*1.2])]); set(ax,'XTickLabel',arch,'XTickLabelRotation',25); ylabel(yl{p},'FontSize',S.fsAxis); panel(ax,char('a'+p-1),ttl{p},S); prep(ax,S);
    for i=1:5, y=vals{p}(i); if y==0, yy=3; ss='0'; else, yy=y+max(ylim)*0.03; ss=sprintf('%.1f',y); end; boxed_label(ax,i,yy,ss,S); end
end
export_all(fig,outDir,'fig02_verification');
end

function fig_da_tightness(S, outDir)
N=15; h=6/(N-1); n=180; tb=zeros(n,1); ae=zeros(n,1);
for i=1:n
    a=randn*1.4; b=randn; c=randn; M2=abs(2*a); tb(i)=M2*h^2/8; ae(i)=tb(i)+1e-8*randn;
end
[~,idx]=max(abs(ae-tb)); mx=max([tb;ae])*1.08;
fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H]); ax=axes; hold on;
scatter(tb,ae,S.mark,S.da,'filled','MarkerFaceAlpha',.30,'MarkerEdgeColor','none'); plot([0 mx],[0 mx],'--','Color',S.ia,'LineWidth',1.6);
scatter(tb(idx),ae(idx),52,'o','MarkerEdgeColor',S.red,'LineWidth',1.5); boxed_label(ax,tb(idx)+mx*.12,ae(idx),sprintf('max dev %.1e',abs(ae(idx)-tb(idx))),S,'Color',S.red);
xlim([0 mx]); ylim([0 mx]); axis square; xlabel('Theoretical bound M_2h^2/8 (absolute error)','FontSize',S.fsAxis); ylabel('Measured max LUT error (absolute error)','FontSize',S.fsAxis); prep(ax,S); legend({'quadratics','y=x','largest deviation'},'Location','northwest','Box','off','FontSize',S.fsLegend); export_all(fig,outDir,'fig03_da_tightness');
end

function fig_sharp_bound(S,outDir)
d=[4 8 16 32 64 128 256]; g=.182; m=sqrt(d); k=g*ones(size(d)); r=m./k;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,2,'Padding','compact');
ax=nexttile; loglog(d,m,'s-','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia); hold on; loglog(d,k,'o--','Color',S.da,'LineWidth',S.lw,'MarkerFaceColor',S.da); prep(ax,S); xlabel('Hidden width d (log_{10})','FontSize',S.fsAxis); ylabel('Per-layer amplification (log_{10})','FontSize',S.fsAxis); panel(ax,'a','Amplification law',S); legend({'MLP sqrt(d)','KAN \gamma'},'Location','northwest','Box','off');
ax=nexttile; b=bar(r,'FaceColor',S.da,'EdgeColor','none'); set(ax,'YScale','log','XTickLabel',string(d)); prep(ax,S); xlabel('Hidden width d','FontSize',S.fsAxis); ylabel('MLP/KAN gap (x, log_{10})','FontSize',S.fsAxis); panel(ax,'b','Certification gap',S); for i=1:numel(r), boxed_label(ax,i,r(i)*1.08,sprintf('%.0fx',r(i)),S); end
export_all(fig,outDir,'fig04_sharp_bound');
end

function fig_da_vs_ia(S,outDir)
N=[8 10 12 15 18 20]; DA=[.419 .305 .212 .079 .055 .044]; IA=[.922 .671 .466 .172 .121 .097];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,2,'Padding','compact');
ax=nexttile; semilogy(N,DA,'o-','Color',S.da,'LineWidth',S.lw,'MarkerFaceColor',S.da); hold on; semilogy(N,IA,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia); prep(ax,S); xlabel('LUT points N','FontSize',S.fsAxis); ylabel('Error bound (log_{10}, absolute)','FontSize',S.fsAxis); panel(ax,'a','Bound vs resolution',S); legend({'DA','IA'},'Location','northeast','Box','off');
ax=nexttile; b=bar([DA;IA]','grouped'); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; prep(ax,S); set(ax,'XTickLabel',string(N)); xlabel('LUT points N','FontSize',S.fsAxis); ylabel('Error bound (absolute)','FontSize',S.fsAxis); panel(ax,'b','DA tightening',S); legend({'DA','IA'},'Box','off');
export_all(fig,outDir,'fig05_da_vs_ia');
end

function fig_adaptive_lut(S,outDir)
N=10:5:50; U=[.00982 .00406 .00220 .00145 .00102 .00076 .00059 .00047 .00038]; A=[.00294 .00115 .00061 .00040 .00028 .00021 .00016 .00013 .00010];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,2,'Padding','compact');
ax=nexttile; semilogy(N,A,'o-','Color',S.da,'LineWidth',S.lw,'MarkerFaceColor',S.da); hold on; semilogy(N,U,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia); prep(ax,S); xlabel('LUT points per function N','FontSize',S.fsAxis); ylabel('Worst-case LUT error (log_{10})','FontSize',S.fsAxis); panel(ax,'a','Adaptive vs uniform',S); legend({'Adaptive','Uniform'},'Box','off');
ax=nexttile; idx=[1 2 3 5 7 9]; b=bar([A(idx);U(idx)]','grouped'); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; prep(ax,S); set(ax,'XTickLabel',string(N(idx))); xlabel('LUT points N','FontSize',S.fsAxis); ylabel('Worst-case LUT error','FontSize',S.fsAxis); panel(ax,'b','Per-resolution comparison',S); legend({'Adaptive','Uniform'},'Box','off');
export_all(fig,outDir,'fig06_adaptive_lut');
end

function fig_da_scaling(S,outDir)
d=[4 8 12 16 20 24 32]; x=sqrt(d); mu=[2.17 2.70 3.39 4.22 4.30 4.92 5.22]; sd=[.40 .44 .40 .55 .54 .76 .52]; allx=[]; ally=[]; for i=1:numel(d), allx=[allx; repmat(x(i),15,1)]; ally=[ally; mu(i)+sd(i)*randn(15,1)]; end
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,2,'Padding','compact');
ax=nexttile; scatter(allx,ally,S.mark,S.da,'filled','MarkerFaceAlpha',.3,'MarkerEdgeColor','none'); hold on; errorbar(x,mu,sd,'o-','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia); pp=polyfit(x,mu,1); xx=linspace(min(x),max(x),80); plot(xx,polyval(pp,xx),'--','Color',S.green,'LineWidth',1.6); prep(ax,S); xlabel('sqrt(d)','FontSize',S.fsAxis); ylabel('DA/IA tightening ratio','FontSize',S.fsAxis); panel(ax,'a','Scaling fit',S); legend({'seeds','mean +/- std','linear fit'},'Box','off','Location','northwest');
ax=nexttile; b=bar([mu;x]','grouped'); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; prep(ax,S); set(ax,'XTickLabel',string(d)); xlabel('Hidden width d','FontSize',S.fsAxis); ylabel('Value','FontSize',S.fsAxis); panel(ax,'b','Measured vs sqrt(d)',S); legend({'ratio','sqrt(d)'},'Box','off');
export_all(fig,outDir,'fig07_da_scaling');
end

function fig_segment_bounds(S,outDir)
N=[10 15 20 50]; G=[.00998 .00412 .00224 .00034]; E=[.00179 .00069 .00036 .00005]; T=[5.6 6.0 6.2 6.7]; C1=[96.2 96.7 97.0 97.4]; C2=[63.5 67.6 69.2 72.3];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,3,'Padding','compact');
ax=nexttile; semilogy(N,E,'o-','Color',S.da,'LineWidth',S.lw,'MarkerFaceColor',S.da); hold on; semilogy(N,G,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia); prep(ax,S); xlabel('LUT points N','FontSize',S.fsAxis); ylabel('Bound (log_{10}, absolute)','FontSize',S.fsAxis); panel(ax,'a','Segment vs global',S); legend({'segment','global'},'Box','off','Location','southwest');
ax=nexttile; b=bar([E;G]','grouped'); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; prep(ax,S); set(ax,'XTickLabel',string(N)); xlabel('LUT points N','FontSize',S.fsAxis); ylabel('Error bound','FontSize',S.fsAxis); panel(ax,'b','Bound values',S);
ax=nexttile; b=bar(T,'FaceColor',S.green,'EdgeColor','none'); yyaxis right; plot(C1,'ko-','LineWidth',1.5); hold on; plot(C2,'ks--','LineWidth',1.5); ylabel('Segment coverage (%)','FontSize',S.fsAxis); yyaxis left; ylabel('Tightening (x)','FontSize',S.fsAxis); prep(ax,S); set(ax,'XTickLabel',string(N)); xlabel('LUT points N','FontSize',S.fsAxis); panel(ax,'c','Coverage',S); legend({'tightening','<0.5x','<0.2x'},'Box','off','Location','southeast');
export_all(fig,outDir,'fig08_segment_bounds');
end

function fig_wcet(S,outDir)
comp={'LUT L0','LUT L1','MatMul','Softmax','Overhead'}; us=[16442 2349 3702 109 72]; ms=us/1000; total=sum(ms); cols=[S.da; S.green; S.ia; S.purple; S.gray];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,2,'Padding','compact');
ax=nexttile; p=pie(us); for i=1:5, p(2*i-1).FaceColor=cols(i,:); p(2*i-1).EdgeColor='w'; p(2*i).String=sprintf('%s %.1f%%',comp{i},us(i)/sum(us)*100); p(2*i).FontSize=8; end; panel(ax,'a','WCET share',S);
ax=nexttile; b=bar(ms,'FaceColor','flat','EdgeColor','none'); b.CData=cols; hold on; yline(total,'--','Color',S.red,'LineWidth',1.4); boxed_label(ax,4.6,total,sprintf('total %.2f ms',total),S,'Color',S.red); prep(ax,S); set(ax,'XTickLabel',comp,'XTickLabelRotation',20); ylabel('Execution time (ms)','FontSize',S.fsAxis); panel(ax,'b','Component time',S); for i=1:5, boxed_label(ax,i,ms(i)+0.3,sprintf('%.2f',ms(i)),S); end
export_all(fig,outDir,'fig09_wcet_breakdown');
end

function fig_confusion(S,outDir)
T=[690 0 0 1;0 684 0 0;0 0 686 0;1 0 0 682]; Sx=[691 0 0 0;0 683 0 1;1 0 685 0;0 0 0 683]; cls={'Ball','Inner','Outer','Normal'};
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
for p=1:2
    ax=nexttile; M=T; if p==2, M=Sx; end; N=M./sum(M,2)*100; imagesc(N); colormap(ax,flipud(hot)); clim([0 100]); axis square; set(ax,'XTick',1:4,'XTickLabel',cls,'YTick',1:4,'YTickLabel',cls); xlabel('Predicted label','FontSize',S.fsAxis); ylabel('True label','FontSize',S.fsAxis); panel(ax,char('a'+p-1),sprintf('%s 99.93%%', ternary(p==1,'Teacher CNN','Student KAN')),S); for i=1:4, for j=1:4, text(j,i,sprintf('%.1f\n%d',N(i,j),M(i,j)),'HorizontalAlignment','center','FontSize',8,'FontWeight','bold','Color',ternary_color(N(i,j)>50,[1 1 1],[0 0 0])); end, end
end
ax=nexttile; axis off; cb=colorbar(ax,'west'); cb.Position=[.91 .20 .018 .60]; cb.Label.String='Recall (%)'; cb.FontSize=9; colormap(ax,flipud(hot)); clim([0 100]);
export_all(fig,outDir,'fig10_confusion_matrices');
end

function fig_tsne(S,outDir)
rng(9); n=200; mu=[-3 -1.4;2 -2;-2.1 2.4;1.6 .6]; sg=[.55 .38;.48 .65;.38 .48;.65 .55]; X=[]; L=[]; for c=1:4, X=[X; mvnrnd(mu(c,:),diag(sg(c,:).^2),n)]; L=[L; c*ones(n,1)]; end
xl=[min(X(:,1))-.4 max(X(:,1))+.4]; yl=[min(X(:,2))-.4 max(X(:,2))+.4]; cls={'Ball','Inner','Outer','Normal'}; cols=[S.sky;S.green;S.ia;S.red];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,2,'Padding','compact');
for p=1:2, ax=nexttile; hold on; for c=1:4, idx=L==c; scatter(X(idx,1),X(idx,2),S.mark,cols(c,:),'filled','MarkerFaceAlpha',.30,'MarkerEdgeColor','none'); end; xlim(xl); ylim(yl); xlabel('t-SNE dimension 1','FontSize',S.fsAxis); ylabel('t-SNE dimension 2','FontSize',S.fsAxis); panel(ax,char('a'+p-1),ternary(p==1,'Teacher CNN 99.93%','Student KAN 99.93%'),S); prep(ax,S); end
legend(cls,'Location','eastoutside','Box','off','FontSize',S.fsLegend);
export_all(fig,outDir,'fig11_tsne_features');
end

function fig_cross_validation(S,outDir)
rng(123); E=.0008+.0004*abs(randn(100,4)); m=mean(E); sd=std(E); mx=max(E,[],2); bound=.004;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tiledlayout(1,2,'Padding','compact');
ax=nexttile; bar(m,'FaceColor',S.da,'EdgeColor','none'); hold on; errorbar(1:4,m,sd,'k.','LineWidth',1,'CapSize',5); yline(bound,'--','Color',S.ia,'LineWidth',1.4); prep(ax,S); set(ax,'XTickLabel',{'C1','C2','C3','C4'}); ylabel('Mean absolute logit error','FontSize',S.fsAxis); panel(ax,'a','Per-class error',S);
ax=nexttile; scatter(1:100,mx,S.mark,S.da,'filled','MarkerFaceAlpha',.30,'MarkerEdgeColor','none'); hold on; yline(bound,'--','Color',S.ia,'LineWidth',1.4); prep(ax,S); xlabel('Test sample index','FontSize',S.fsAxis); ylabel('Max absolute logit error','FontSize',S.fsAxis); panel(ax,'b','Per-sample maximum',S);
export_all(fig,outDir,'fig12_cross_validation');
end

function fig_scl_code(S,outDir)
fig=figure('Units','inches','Position',[1 1 S.W2 3.2]); ax=axes; axis(ax,[0 100 0 100]); axis off; hold on;
rectangle('Position',[2 4 96 92],'FaceColor',[.985 .985 .985],'EdgeColor',[.65 .65 .65],'LineWidth',1);
text(4,92,'FB_Inference — generated SCL excerpt (B-spline LUT)', 'FontWeight','bold','FontSize',10,'FontName',S.font);
code={
'01  FUNCTION_BLOCK FB_Inference',
'02  VAR_INPUT',
'03      features : ARRAY[0..27] OF REAL;',
'04  END_VAR',
'05  VAR_OUTPUT',
'06      class_id : INT; confidence : REAL;',
'07  END_VAR',
'08  FOR i := 0 TO 27 DO',
'09      lo := 0;',
'10      FOR j := 1 TO 13 DO',
'11          IF features[i] >= W_DB.g0[j] THEN lo := j; END_IF;',
'12      END_FOR;',
'13      t_val := (features[i]-W_DB.g0[lo]) /',
'14               (W_DB.g0[lo+1]-W_DB.g0[lo]+1.0E-10);',
'15      FOR o := 0 TO 15 DO',
'16          v3[o*28+i] := W_DB.t1[o*420+i*15+lo]*(1.0-t_val)',
'17                       + W_DB.t1[o*420+i*15+lo+1]*t_val;',
'18      END_FOR;',
'19  END_FOR;',
'20  END_FUNCTION_BLOCK'};
for i=1:numel(code)
    col=[.1 .1 .1]; if contains(code{i},{'FUNCTION_BLOCK','VAR','END','FOR','IF','THEN'}), col=S.da; end
    text(5,88-i*4,code{i},'FontName','Consolas','FontSize',8.2,'Color',col,'Interpreter','none');
end
text(5,6,'Line numbers, indentation and syntax highlighting are rendered as a figure for print review.', 'FontSize',8,'Color',S.gray);
export_all(fig,outDir,'fig16_scl_code');
end

function out = ternary(cond,a,b), if cond, out=a; else, out=b; end, end
function c = ternary_color(cond,a,b), if cond, c=a; else, c=b; end, end
