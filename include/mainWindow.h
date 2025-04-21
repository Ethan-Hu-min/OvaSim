#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_mainWindow.h"
#include "SimulatingWidget.h"
#include "HistoryWidget.h"
#include "DeviceWidget.h"
#include "choisewidget.h"
#include "GlobalConfig.h"


class OvaSim : public QMainWindow
{
    Q_OBJECT

public:
    OvaSim(QWidget *parent = nullptr);
    ~OvaSim();
    QString userName;
    QString choiseMode;
    QString choiseSample;

   

private:
    Ui::OvaSimClass ui;

    SimulatingWidget* SimWidget = nullptr;
    ChoiseWidget* CoiWidget = nullptr;
    HistoryWidget* HisWidget = nullptr;
    DeviceWidget* DevWidget = nullptr;



private slots:
    void on_startbutton_clicked();
    void on_historybutton_clicked();
    void on_devicebutton_clicked();
    void choise_widget_close();

};
