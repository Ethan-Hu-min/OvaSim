#pragma once

#include <QtWidgets/QMainWindow>
#include <QOpenGLWidget>
#include "ui_SimulatingWidget.h"
#include <QTimer>

#include <GLWidget.h>
#include <unordered_set>



class SimulatingWidget : public QWidget
{
    Q_OBJECT

public:
    SimulatingWidget(QWidget* parent = nullptr);
    ~SimulatingWidget();
    void closeEvent(QCloseEvent*);
    void keyPressEvent(QKeyEvent* event);
    void keyReleaseEvent(QKeyEvent* event);

    HardwareInfo SWHWinfo;

signals:
    void ExitWin();

private:
    Ui::SimulatingWidgetClass ui;
    GLWidget* GLDisplayWidget = nullptr;
    QTimer* timer = nullptr;
    QTimer* operateTimer = nullptr;
    bool isRunning = false;
    QDateTime  startTime;
    std::unordered_set<int> errorRecord;

private slots:
    void startTimer();
    void endTimer();
    void displayTimer();
    void displayError(int infoType);

};