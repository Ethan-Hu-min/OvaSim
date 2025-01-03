#pragma once

#include <QtWidgets/QMainWindow>
#include <QOpenGLWidget>
#include "ui_SimulatingWidget.h"
#include <QTimer>

#include <GLWidget.h>


class SimulatingWidget : public QWidget
{
    Q_OBJECT

public:
    SimulatingWidget(QWidget* parent = nullptr);
    ~SimulatingWidget();
    void closeEvent(QCloseEvent*);

signals:
    void ExitWin();

private:
    Ui::SimulatingWidgetClass ui;
    GLWidget* GLDisplayWidget = nullptr;
    QTimer* timer;

private slots:
    void displayImage();

};