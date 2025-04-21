#include "mainWindow.h"





OvaSim::OvaSim(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);


    connect(ui.StartButton, SIGNAL(clicked(bool)), this, SLOT(on_startbutton_clicked()));
    connect(ui.HistoryButton, SIGNAL(clicked(bool)), this, SLOT(on_historybutton_clicked()));
    connect(ui.DeviceButton, SIGNAL(clicked(bool)), this, SLOT(on_devicebutton_clicked()));
    connect(ui.QuitButton, SIGNAL(clicked(bool)), this, SLOT(close()));

    //QDialog* dialog = new QDialog(this);
    //QVBoxLayout* dialogLayout = new QVBoxLayout(dialog);
    //QFont dialogFont = dialog->font();
    //dialogFont.setPointSize(20); // 可根据需求调整大小
    //dialog->setFont(dialogFont);
    //QLabel* label = new QLabel("你的操作评价为：     优秀.", dialog);
    //dialogLayout->addWidget(label);

    // 连接按钮的 clicked 信号到槽函数
    //connect(ui.QuitButton, SIGNAL(clicked(bool)), dialog, SLOT(show()));

}

OvaSim::~OvaSim()
{}

void OvaSim::on_startbutton_clicked() {
    this->close();
    CoiWidget = new ChoiseWidget;
    CoiWidget->show();
    connect(CoiWidget, SIGNAL(ExitWin()), this, SLOT(choise_widget_close()));
}

void OvaSim::choise_widget_close() {
    CoiWidget->getInfo(userName, choiseSample, choiseMode);

    qDebug() << "userName:" << userName;
    qDebug() << "choiseSample:" << choiseSample;
    qDebug() << "choiseMode:" << choiseMode;

    CoiWidget->close();
    SimWidget = new SimulatingWidget;
    SimWidget->show();
    connect(SimWidget, SIGNAL(ExitWin()), this, SLOT(show()));
}

void OvaSim::on_historybutton_clicked() {
    this->close();
    HisWidget = new HistoryWidget;
    HisWidget->show();
    connect(HisWidget, SIGNAL(ExitWin()), this, SLOT(show()));
}
void OvaSim::on_devicebutton_clicked() {
    this->close();
    DevWidget = new DeviceWidget;
    DevWidget->show();
    connect(DevWidget, SIGNAL(ExitWin()), this, SLOT(show()));
}
