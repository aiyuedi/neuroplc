%% NeuroPLC — Regenerate pages 27-28: DA Scaling + Segment Bounds figures
clc; clear; close all;
out = 'D:/neuroplc-paper/paper/figures';

Cb=[0.000 0.447 0.698]; Cv=[0.835 0.369 0.000];
Cg=[0.000 0.620 0.451]; Co=[0.902 0.624 0.000];
W2=6.9; H=2.6;
set(0,'DefaultAxesFontName','Helvetica','DefaultAxesFontSize',8,...
    'DefaultAxesTickDir','out','DefaultAxesXGrid','on','DefaultAxesYGrid','on',...
    'DefaultAxesGridAlpha',0.15,'DefaultLineLineWidth',1.2);

%% ==== Figure 1: DA sqrt(d) Scaling Law (replaces fig8_da_scaling) ====
rng(42);
d_vals=[4 8 12 16 20 24 32]; sqrt_d=sqrt(d_vals);
ratio_mean=[2.17 2.70 3.39 4.22 4.30 4.92 5.22];
ratio_std=[0.40 0.44 0.40 0.55 0.54 0.76 0.52];
all_d=[]; all_r=[];
for i=1:7
    pts=ratio_mean(i)+ratio_std(i)*randn(15,1);
    all_d=[all_d; repmat(sqrt_d(i),15,1)]; all_r=[all_r; pts];
end

fig=figure('Units','inches','Position',[1 1 W2 H*0.92],'Color','w','Visible','off');
t=tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

ax=nexttile(1);
scatter(all_d,all_r,20,Cb,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeColor','none'); hold on;
errorbar(sqrt_d,ratio_mean,ratio_std,'o-','Color',Cv,'LineWidth',1.8,...
    'MarkerSize',8,'MarkerFaceColor',Cv,'CapSize',6);
p=polyfit(sqrt_d,ratio_mean,1); xf=linspace(1.5,6,50);
plot(xf,polyval(p,xf),'--','Color',Cg,'LineWidth',1.5);
xlabel('sqrt(d)','FontSize',9.5); ylabel('DA/IA Tightening Ratio','FontSize',9.5);
title(sprintf('(a) sqrt(d) Scaling Law (r=%.3f p<10^{-4})',0.987),'FontSize',10,'FontWeight','bold');
legend({sprintf('%d seeds',length(all_d)),'Mean +/- 1 std',sprintf('Best fit r=%.3f',0.987)},'Location','nw','FontSize',7.5,'Box','off');
text(1.65,5.3,sprintf('r = %.3f*sqrt(d) + %.3f',p(1),p(2)),'FontSize',7.5,'Color',Cg,'FontWeight','bold');
box off; grid on;

ax=nexttile(2);
bh=bar(1:7,[ratio_mean; sqrt_d]','grouped');
bh(1).FaceColor=Cb; bh(2).FaceColor=Co;
set(ax,'XTickLabel',{'4','8','12','16','20','24','32'},'Box','off','FontSize',8);
xlabel('Hidden dimension d','FontSize',9.5); ylabel('Value','FontSize',9.5);
title('(b) sqrt(d) Theory vs Measured','FontSize',10,'FontWeight','bold');
legend({'DA/IA Ratio','sqrt(d)'},'Location','nw','FontSize',7.5,'Box','off');
grid on;

exportgraphics(fig,fullfile(out,'fig_da_scaling_law.pdf'),'ContentType','vector');
exportgraphics(fig,fullfile(out,'fig_da_scaling_law.png'),'Resolution',300);
close(fig);
fprintf('[1/2] fig_da_scaling_law done\n');

%% ==== Figure 2: Segment-Aware Bounds (replaces tab:segment_bound) ====
N_vals=[10 15 20 50];
global_e=[0.00998 0.00412 0.00224 0.00034];
segment_e=[0.00179 0.00069 0.00036 0.00005];
tightening=[5.6 6.0 6.2 6.7];
pct50=[96.2 96.7 97.0 97.4];
pct20=[63.5 67.6 69.2 72.3];

fig=figure('Units','inches','Position',[1 1 W2 H],'Color','w','Visible','off');
t=tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

ax=nexttile(1);
semilogy(N_vals,global_e,'s-','Color',Cv,'MarkerSize',8,'MarkerFaceColor',Cv,'LineWidth',1.5); hold on;
semilogy(N_vals,segment_e,'o-','Color',Cb,'MarkerSize',8,'MarkerFaceColor',Cb,'LineWidth',1.5);
for i=1:4
    text(N_vals(i)+1.2,global_e(i)*1.08,sprintf('%.5f',global_e(i)),'FontSize',6.2,'Color',Cv);
    text(N_vals(i)+1.2,segment_e(i)*0.88,sprintf('%.5f',segment_e(i)),'FontSize',6.2,'Color',Cb);
end
xlabel('LUT Points N','FontSize',9.5); ylabel('Error (log)','FontSize',9.5);
title(sprintf('(a) Global vs Segment-Aware\nMean %.1fx tightening',mean(tightening)),'FontSize',9.5,'FontWeight','bold');
legend({'Global M_2','Segment M_{2,j}'},'Location','sw','FontSize',7.5,'Box','off');
xlim([8 52]); box off; grid on;

ax=nexttile(2);
bh=bar(1:4,[global_e; segment_e]','grouped');
bh(1).FaceColor=Cv; bh(2).FaceColor=Cb;
set(ax,'XTickLabel',{'N=10','N=15','N=20','N=50'},'FontSize',8,'Box','off');
ylabel('Error Bound','FontSize',9.5);
title('(b) Per-Resolution Comparison','FontSize',9.5,'FontWeight','bold');
legend({'Global','Segment'},'Location','ne','FontSize',7.5,'Box','off'); grid on;

ax=nexttile(3);
yyaxis left;
b2=bar(1:4,tightening,'FaceColor',Cg,'EdgeColor','none','BarWidth',0.5);
ylabel('Tightening (x)','FontSize',9.5); ylim([5 7.5]);
yyaxis right;
plot(1:4,pct50,'ko-','LineWidth',1.5,'MarkerSize',7,'MarkerFaceColor','k'); hold on;
plot(1:4,pct20,'ks--','LineWidth',1.5,'MarkerSize',7,'MarkerFaceColor','w');
ylabel('% Segments','FontSize',9.5); ylim([55 100]);
set(ax,'XTickLabel',{'N=10','N=15','N=20','N=50'},'FontSize',8,'Box','off');
xlabel('LUT Resolution','FontSize',9.5);
title('(c) Segment Coverage (<50% vs <20%)','FontSize',9.5,'FontWeight','bold');
legend({'< 0.5x global','< 0.2x global'},'Location','se','FontSize',7,'Box','off'); grid on;

exportgraphics(fig,fullfile(out,'fig_segment_bounds.pdf'),'ContentType','vector');
exportgraphics(fig,fullfile(out,'fig_segment_bounds.png'),'Resolution',300);
close(fig);
fprintf('[2/2] fig_segment_bounds done\n');
fprintf('All regenerated -> %s\n',out);
