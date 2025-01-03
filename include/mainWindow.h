#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_mainWindow.h"



class OvaSim : public QMainWindow
{
    Q_OBJECT

public:
    OvaSim(QWidget *parent = nullptr);
    ~OvaSim();

private:
    Ui::OvaSimClass ui;

private slots:
    void on_startbutton_clicked();
    void on_historybutton_clicked();
    void on_devicebutton_clicked();
};
