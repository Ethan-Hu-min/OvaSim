#include "mainWindow.h"
#include "SimulatingWidget.h"
#include "HistoryWidget.h"
#include "DeviceWidget.h"

#include "GlobalConfig.h"


OvaSim::OvaSim(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    connect(ui.StartButton, SIGNAL(clicked(bool)), this, SLOT(on_startbutton_clicked()));
    connect(ui.HistoryButton, SIGNAL(clicked(bool)), this, SLOT(on_historybutton_clicked()));
    connect(ui.DeviceButton, SIGNAL(clicked(bool)), this, SLOT(on_devicebutton_clicked()));
    connect(ui.QuitButton, SIGNAL(clicked(bool)), this, SLOT(close()));

}

OvaSim::~OvaSim()
{}

void OvaSim::on_startbutton_clicked() {
    this->close();
    SimulatingWidget* SimWidget = new SimulatingWidget;
    SimWidget->show();
    connect(SimWidget, SIGNAL(ExitWin()), this, SLOT(show()));
}
void OvaSim::on_historybutton_clicked() {
    this->close();
    HistoryWidget* HisWidget = new HistoryWidget;
    HisWidget->show();
    connect(HisWidget, SIGNAL(ExitWin()), this, SLOT(show()));
}
void OvaSim::on_devicebutton_clicked() {
    this->close();
    DeviceWidget* DevWidget = new DeviceWidget;
    DevWidget->show();
    connect(DevWidget, SIGNAL(ExitWin()), this, SLOT(show()));
}