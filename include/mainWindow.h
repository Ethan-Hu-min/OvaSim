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
};
